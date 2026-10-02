"""
LangGraph Chatbot Backend (with Tool Calling)
---------------------------------------------
A persistent, streaming chatbot graph built on LangGraph + SQLite checkpointing,
with LangSmith tracing, automated per-response evaluation, and 3 tools:

  1. duckduckgo_search  - web search for current info / facts
  2. get_stock_price    - latest stock quote via Alpha Vantage
  3. calculator         - basic arithmetic (add, sub, mul, div, pow)

Graph:  START -> chat_node -> (tools_condition) -> tools -> chat_node -> ... -> END

Required .env keys:
  OPENAI_API_KEY
  ALPHAVANTAGE_API_KEY          (for the stock tool)
Optional:
  LANGSMITH_TRACING, LANGSMITH_API_KEY, ENABLE_EVALUATION, ENABLE_LLM_JUDGE,
  OPENAI_MODEL, OPENAI_TEMPERATURE, CHATBOT_DB_PATH, CHATBOT_SYSTEM_PROMPT

Install:  pip install -U ddgs langchain-community requests
"""

import os
import sqlite3
import threading
import uuid
from typing import Annotated, List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

load_dotenv()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DB_PATH = os.getenv("CHATBOT_DB_PATH", "chatbot.db")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a helpful, friendly assistant. Answer clearly and concisely.\n"
    "You have 3 tools and should use them whenever relevant:\n"
    "- duckduckgo_search: for current events, news, or any fact you are not "
    "sure about.\n"
    "- get_stock_price: for any stock price / quote question (pass the ticker "
    "symbol, e.g. AAPL, TSLA).\n"
    "- calculator: for ANY arithmetic. Never do math in your head.\n"
    "After using a tool, answer the user using the tool's result.",
)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to a .env file or your environment "
        "before starting the app."
    )

# LangSmith tracing is opt-in and purely additive.
LANGSMITH_ENABLED = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if LANGSMITH_ENABLED and not os.getenv("LANGSMITH_API_KEY"):
    raise EnvironmentError(
        "LANGSMITH_TRACING is set to true but LANGSMITH_API_KEY is missing. "
        "Add it to your .env file, or set LANGSMITH_TRACING=false to disable tracing."
    )

EVALUATION_ENABLED = LANGSMITH_ENABLED and os.getenv(
    "ENABLE_EVALUATION", "true").lower() == "true"
ENABLE_LLM_JUDGE = os.getenv("ENABLE_LLM_JUDGE", "true").lower() == "true"

model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
langsmith_client = Client() if LANGSMITH_ENABLED else None


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div, pow
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        elif operation == "pow":
            result = first_num ** second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "ALPHAVANTAGE_API_KEY is not set in the environment."}
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper().strip(),
                "apikey": ALPHAVANTAGE_API_KEY,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        # Alpha Vantage returns HTTP 200 with a message on rate limit / bad key.
        for key in ("Note", "Information", "Error Message"):
            if key in data:
                return {"error": data[key]}
        if not data.get("Global Quote"):
            return {"error": f"No quote found for symbol '{symbol}'."}
        return data
    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {e}"}


tools = [search_tool, get_stock_price, calculator]
model_with_tools = model.bind_tools(tools)


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
# The judge uses the plain model (no tools bound).
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
    Run evaluators and attach the results as LangSmith feedback on run_id.
    Runs in a background thread; failures are swallowed.
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
        pass


# --------------------------------------------------------------------------- #
# Graph state & nodes
# --------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@traceable(name="chat_node", run_type="chain")
def chat_node(state: ChatState) -> dict:
    """LLM node: either answers directly or requests one or more tool calls.

    Evaluation only runs on FINAL answers (no pending tool calls), since
    tool-call turns have empty text content.
    """
    messages = state["messages"]

    # Inject the system prompt (not persisted in state, added on every call).
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = model_with_tools.invoke(messages)

    if EVALUATION_ENABLED and not response.tool_calls:
        run_tree = get_current_run_tree()
        run_id = run_tree.id if run_tree else None
        last_human = next(
            (m.content for m in reversed(messages)
             if isinstance(m, HumanMessage)), ""
        )
        threading.Thread(
            target=_log_evaluation,
            args=(run_id, last_human, response.content),
            daemon=True,
        ).start()

    return {"messages": [response]}


tool_node = ToolNode(tools)


# --------------------------------------------------------------------------- #
# Persistence (SQLite checkpointer) & graph
# --------------------------------------------------------------------------- #
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)  # -> "tools" or END
graph.add_edge("tools", "chat_node")

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
    """Return the full message history for a given thread (includes tool messages)."""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state else []


def get_thread_title(thread_id: str, max_len: int = 40) -> str:
    """Build a sidebar label from the first user message in a thread."""
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content:
            text = " ".join(msg.content.split())
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return "New conversation"


def delete_thread(thread_id: str) -> None:
    """
    Remove a conversation from the checkpoint database.
    Failures on any single table are swallowed (schema varies by version).
    """
    cur = conn.cursor()
    for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
        try:
            cur.execute(
                f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()
