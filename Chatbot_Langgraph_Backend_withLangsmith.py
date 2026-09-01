"""
LangGraph Chatbot Backend
-------------------------
A persistent, streaming chatbot graph built on LangGraph + SQLite checkpointing,
with LangSmith tracing and automated per-response evaluation.

Improvements over the base version:
  - Config driven by environment variables (model, temperature, system prompt, db path)
  - Startup validation for the OpenAI API key
  - Configurable system prompt injected once per thread
  - Thread titles derived from the first user message (for a nicer sidebar)
  - delete_thread() to remove a conversation from the database
  - LangSmith tracing via @traceable (opt-in, no-op if disabled)
  - Automated evaluation of every AI response, logged as LangSmith feedback:
      - Fast heuristic checks (non-empty, reasonable length)
      - An LLM-as-judge score for helpfulness/relevance (optional, toggleable)
    Both show up attached to the run in the LangSmith UI, right next to the trace.
  - Type hints and docstrings throughout
"""

import os
import sqlite3
import threading
import uuid
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Chatbot'


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

# LangSmith tracing is opt-in and purely additive: if LANGSMITH_TRACING /
# LANGSMITH_API_KEY aren't set, @traceable becomes a no-op and the app
# behaves exactly as before. There's no hard requirement on LangSmith to run.
LANGSMITH_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if LANGSMITH_ENABLED and not os.getenv("LANGSMITH_API_KEY"):
    raise EnvironmentError(
        "LANGSMITH_TRACING is set to true but LANGSMITH_API_KEY is missing. "
        "Add it to your .env file, or set LANGSMITH_TRACING=false to disable tracing."
    )

# Evaluation only makes sense if tracing is on (nowhere to see the scores otherwise).
EVALUATION_ENABLED = LANGSMITH_ENABLED and os.getenv(
    "ENABLE_EVALUATION", "true").lower() == "true"
# The LLM-judge adds one extra model call per message (small cost + latency).
# Turn it off and keep only the free heuristic checks if you want zero overhead.
ENABLE_LLM_JUDGE = os.getenv("ENABLE_LLM_JUDGE", "true").lower() == "true"

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
langsmith_client = Client() if LANGSMITH_ENABLED else None


# --------------------------------------------------------------------------- #
# Evaluation: heuristics + LLM-as-judge
# --------------------------------------------------------------------------- #
class ResponseJudgement(BaseModel):
    helpfulness: float = Field(
        description="Score from 0.0 to 1.0 for how helpful and relevant the "
        "response is to the user's message."
    )
    reasoning: str = Field(
        description="One short sentence justifying the score.")


_judge_parser = PydanticOutputParser(pydantic_object=ResponseJudgement)
_judge_prompt = PromptTemplate(
    template=(
        "You are grading an AI assistant's reply for helpfulness and relevance.\n\n"
        "User message:\n{user_message}\n\n"
        "AI response:\n{ai_response}\n\n"
        "{format_instructions}"
    ),
    input_variables=["user_message", "ai_response"],
    partial_variables={
        "format_instructions": _judge_parser.get_format_instructions()},
)
_judge_chain = _judge_prompt | model | _judge_parser


@traceable(name="llm_judge_evaluator", run_type="chain")
def _llm_judge(user_message: str, ai_response: str) -> ResponseJudgement:
    """Score a response's helpfulness using the same LLM as a judge."""
    return _judge_chain.invoke({"user_message": user_message, "ai_response": ai_response})


def _heuristic_scores(ai_response: str) -> List[tuple]:
    """Fast, free, no-LLM-call checks. Each item: (key, score 0-1, comment)."""
    text = ai_response.strip()
    word_count = len(text.split())

    non_empty_score = 1.0 if text else 0.0
    length_ok_score = 1.0 if 3 <= word_count <= 400 else 0.5

    return [
        ("non_empty", non_empty_score, f"{word_count} words"),
        ("reasonable_length", length_ok_score, f"{word_count} words"),
    ]


def _log_evaluation(run_id, user_message: str, ai_response: str) -> None:
    """
    Run evaluators and attach the results as LangSmith feedback on the given
    run_id. Runs in a background thread so it never adds latency to the
    user-facing response. Any failure here is swallowed -- evaluation must
    never break the chat itself.
    """
    if not (EVALUATION_ENABLED and langsmith_client is not None and run_id is not None):
        return

    try:
        for key, score, comment in _heuristic_scores(ai_response):
            langsmith_client.create_feedback(
                run_id=run_id, key=key, score=score, comment=comment)

        if ENABLE_LLM_JUDGE:
            judgement = _llm_judge(user_message, ai_response)
            langsmith_client.create_feedback(
                run_id=run_id,
                key="llm_judge_helpfulness",
                score=judgement.helpfulness,
                comment=judgement.reasoning,
            )
    except Exception:
        # Evaluation is best-effort observability, not core app functionality.
        pass


# --------------------------------------------------------------------------- #
# Graph state & node
# --------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@traceable(name="chat_node", run_type="chain")
def chat_node(state: ChatState) -> dict:
    """Send the running conversation to the LLM, return its reply, and kick
    off async evaluation of that reply.

    Decorated with @traceable so each invocation shows up as a named run in
    LangSmith (when LANGSMITH_TRACING=true). The underlying model.invoke()
    call is also auto-traced as a nested LLM run, so you get the full
    conversation + latency + token usage in one trace tree -- and, if
    evaluation is enabled, feedback scores attached to that same run.
    """
    messages = state["messages"]

    # Inject the system prompt once, at the very start of the conversation.
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = model.invoke(messages)

    if EVALUATION_ENABLED:
        run_tree = get_current_run_tree()
        run_id = run_tree.id if run_tree else None
        last_human = next(
            (m.content for m in reversed(messages)
             if isinstance(m, HumanMessage)), ""
        )
        # Fire-and-forget: don't make the user wait on evaluation.
        threading.Thread(
            target=_log_evaluation,
            args=(run_id, last_human, response.content),
            daemon=True,
        ).start()

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
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state else []


def get_thread_title(thread_id: str, max_len: int = 40) -> str:
    """
    Build a human-friendly sidebar label from the first user message in a
    thread, so the UI doesn't have to show raw UUIDs.
    """
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content:
            # collapse whitespace/newlines
            text = " ".join(msg.content.split())
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
            cur.execute(
                f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
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
