"""
LangGraph Chatbot Backend
-------------------------
A persistent, streaming chatbot graph built on LangGraph + SQLite checkpointing.

Improvements over the base version:
  - Config driven by environment variables (model, temperature, system prompt, db path)
  - Startup validation for the OpenAI API key
  - Configurable system prompt injected once per thread
  - Thread titles derived from the first user message (for a nicer sidebar)
  - delete_thread() to remove a conversation from the database
  - Type hints and docstrings throughout
"""

import os
import sqlite3
import uuid
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DB_PATH = os.getenv("CHATBOT_DB_PATH", "chatbot.db")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a helpful, friendly assistant. Answer clearly and concisely.",
)

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to a .env file or your environment "
        "before starting the app."
    )

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# --------------------------------------------------------------------------- #
# Graph state & node
# --------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """Send the running conversation to the LLM and return its reply."""
    messages = state["messages"]

    # Inject the system prompt once, at the very start of the conversation.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = model.invoke(messages)
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Persistence (SQLite checkpointer)
# --------------------------------------------------------------------------- #
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Helper functions used by the frontend
# --------------------------------------------------------------------------- #
def generate_thread_id() -> str:
    """Create a fresh, unique thread/conversation id."""
    return str(uuid.uuid4())


def retrieve_all_threads() -> List[str]:
    """Return every distinct thread_id currently stored in the checkpoint DB."""
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)


def load_conversation(thread_id: str) -> List[BaseMessage]:
    """Return the full message history for a given thread."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state else []


def get_thread_title(thread_id: str, max_len: int = 40) -> str:
    """
    Build a human-friendly sidebar label from the first user message in a
    thread, so the UI doesn't have to show raw UUIDs.
    """
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content:
            text = " ".join(msg.content.split())  # collapse whitespace/newlines
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return "New conversation"


def delete_thread(thread_id: str) -> None:
    """
    Remove a conversation from the checkpoint database.

    Note: table names come from langgraph's SqliteSaver schema and can vary
    slightly across langgraph versions, so failures on any single table are
    swallowed rather than raising.
    """
    cur = conn.cursor()
    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        try:
            cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()


"""
Streaming reference (used in the Streamlit frontend):

for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content='What is the recipe to make pasta')]},
        config={'configurable': {'thread_id': 'thread-1'}},
        stream_mode='messages'):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
"""
