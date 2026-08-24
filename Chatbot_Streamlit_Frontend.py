"""
LangGraph Chatbot Frontend (Streamlit)
---------------------------------------
Improvements over the base version:
  - Polished page config, custom CSS, and a branded header/footer
  - Sidebar shows readable conversation titles (not raw UUIDs), newest first
  - Per-conversation delete button
  - Robust error handling around streaming so a failed API call doesn't crash the app
  - Markdown rendering for AI responses (code blocks, lists, etc. render properly)
  - "Thinking…" spinner while waiting for the first token
"""

import streamlit as st
from langchain_core.messages import HumanMessage

from Chatbot_Langgraph_Backend import (
    chatbot,
    delete_thread,
    generate_thread_id,
    get_thread_title,
    load_conversation,
    retrieve_all_threads,
)

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
    messages = load_conversation(thread_id)
    st.session_state["message_history"] = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages
    ]
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

    st.markdown('<div class="app-footer">Built with LangGraph + Streamlit</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here…")

if user_input:
    add_thread(st.session_state["thread_id"])

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking…"):
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if message_chunk.content:
                        full_response += message_chunk.content
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as exc:  # noqa: BLE001 - surface any API/streaming failure to the user
            full_response = f"⚠️ Something went wrong while generating a response: {exc}"
            placeholder.error(full_response)

    st.session_state["message_history"].append({"role": "assistant", "content": full_response})
