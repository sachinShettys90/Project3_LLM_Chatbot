"""
LangGraph Chatbot Frontend (Streamlit) - with Tool Calling
-----------------------------------------------------------
  - Streams only the assistant's text (tool-call chunks and raw ToolMessages are ignored)
  - Shows which tool is being used (search / stock / calculator) while it runs
  - Saved history hides tool plumbing (ToolMessages and empty tool-call AI messages)
  - Readable sidebar titles, per-conversation delete, error handling, markdown rendering
"""

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from Chatbot_Langgraph_Backend_withLangsmith import (
    chatbot,
    delete_thread,
    generate_thread_id,
    get_thread_title,
    load_conversation,
    retrieve_all_threads,
)

TOOL_LABELS = {
    "duckduckgo_search": "🔎 Web search",
    "get_stock_price": "📈 Stock price",
    "calculator": "🧮 Calculator",
}


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, f"🔧 {name}")


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 12px; }
        div[data-testid="stSidebarUserContent"] button { text-align: left; }
        footer { visibility: hidden; }
        .app-footer {
            position: fixed; bottom: 0; left: 0; right: 0;
            text-align: center; font-size: 0.75rem; color: gray;
            padding: 6px 0; background: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def reset_chat() -> None:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def switch_thread(thread_id: str) -> None:
    """Load a saved thread, keeping only user messages and final AI answers.
    Tool names used for each answer are re-attached so the badge still shows."""
    history = []
    pending_tools: list[str] = []
    for m in load_conversation(thread_id):
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content, "tools": []})
            pending_tools = []
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                pending_tools.extend(tc["name"] for tc in m.tool_calls)
            elif m.content:
                history.append(
                    {"role": "assistant", "content": m.content,
                     "tools": pending_tools})
                pending_tools = []
        # ToolMessage -> skipped
    st.session_state["message_history"] = history
    st.session_state["thread_id"] = thread_id


def remove_thread(thread_id: str) -> None:
    delete_thread(thread_id)
    st.session_state["chat_threads"].remove(thread_id)
    if st.session_state["thread_id"] == thread_id:
        reset_chat()


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("💬 LangGraph Chatbot")
    st.button("➕ New chat", on_click=reset_chat, use_container_width=True)
    st.header("My Conversations")

    for thread_id in st.session_state["chat_threads"][::-1]:
        title = get_thread_title(thread_id)
        is_active = thread_id == st.session_state["thread_id"]
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                ("🟢 " if is_active else "") + title,
                key=f"open-{thread_id}",
                use_container_width=True,
            ):
                switch_thread(thread_id)
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-{thread_id}"):
                remove_thread(thread_id)
                st.rerun()

    st.markdown('<div class="app-footer">Built with LangGraph + Streamlit</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message.get("tools"):
            st.caption("Used: " + ", ".join(
                dict.fromkeys(tool_label(t) for t in message["tools"])))
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask anything — I can search the web, check stocks, and calculate…")

if user_input:
    add_thread(st.session_state["thread_id"])

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input, "tools": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "tags": [f"thread:{st.session_state['thread_id']}"],
        "metadata": {"thread_id": st.session_state["thread_id"]},
    }

    tools_used: list[str] = []

    with st.chat_message("assistant"):
        tool_slot = st.empty()      # shows tool activity above the answer
        placeholder = st.empty()    # streams the answer text
        full_response = ""
        try:
            with st.spinner("Thinking…"):
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # 1) LLM decided to call a tool -> show which one
                    if isinstance(message_chunk, AIMessageChunk):
                        for tc in message_chunk.tool_call_chunks or []:
                            name = tc.get("name")
                            if name and name not in tools_used:
                                tools_used.append(name)
                                tool_slot.caption(
                                    "Using: " + ", ".join(
                                        tool_label(t) for t in tools_used) + " …")

                        # 2) Normal answer text (skip empty tool-call chunks)
                        if (
                            metadata.get("langgraph_node") == "chat_node"
                            and isinstance(message_chunk.content, str)
                            and message_chunk.content
                        ):
                            full_response += message_chunk.content
                            placeholder.markdown(full_response + "▌")

                    # 3) ToolMessage (raw tool output) -> intentionally not displayed
                    elif isinstance(message_chunk, ToolMessage):
                        continue

            placeholder.markdown(full_response)
            if tools_used:
                tool_slot.caption(
                    "Used: " + ", ".join(tool_label(t) for t in tools_used))
        except Exception as exc:  # noqa: BLE001
            full_response = f"⚠️ Something went wrong while generating a response: {exc}"
            placeholder.error(full_response)

    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response, "tools": tools_used})
