# LLM Chatbot with LangGraph & Streamlit

A conversational chatbot built using **LangGraph** (for stateful, graph-based LLM orchestration) and **Streamlit** (for the chat UI). The project progresses through three stages — a stateless chatbot, a memory-enabled chatbot, and a persistence deep-dive — before wiring the memory-enabled version into a web frontend.

## Project Structure

```
.
├── 1_BasicChatbot.ipynb                 # Stage 1: Stateless chatbot (no memory)
├── 2_BasicChatbot_withMemory.ipynb      # Stage 2: Adds checkpointed memory (per thread)
├── 3_Persistence.ipynb                  # Concept notebook: LangGraph persistence features
├── Chatbot_Langgraph_Backend.py         # Compiled LangGraph chatbot (used by the Streamlit app)
└── 5_Chatbot_Streamlit_Frontend.py      # Streamlit chat UI
```

## Overview

### 1. `1_BasicChatbot.ipynb` — Basic Chatbot
- Builds a single-node LangGraph (`START → chat_node → END`) wrapping `ChatOpenAI`.
- Each call to `chatbot.invoke()` is stateless — no memory of prior turns.
- Includes a simple CLI loop (`input()` / `print()`) to chat until the user types `quit`/`exit`/`bye`.
- **Limitation (documented in the notebook itself):** every invocation is a fresh call to the LLM with no chat history, motivating the next stage.

### 2. `2_BasicChatbot_withMemory.ipynb` — Chatbot with Memory
- Same graph structure as Stage 1, but compiles with a `MemorySaver` checkpointer:
  ```python
  checkpointer = MemorySaver()
  chatbot = graph.compile(checkpointer=checkpointer)
  ```
- Introduces `thread_id` in the invocation config so conversation history is tracked per session/thread:
  ```python
  config = {'configurable': {'thread_id': thread_id}}
  chatbot.invoke({'messages': [...]}, config=config)
  ```
- Uses `chatbot.get_state(config)` to inspect the accumulated state for a given thread.
- This is the version of the chatbot the Streamlit frontend is built to use.

### 3. `Chatbot_Langgraph_Backend.py` — Backend Module
- Standalone `.py` version of the Stage 2 chatbot graph, structured so it can be imported rather than run top-to-bottom in a notebook.
- Same `ChatState` / `chat_node` / graph wiring as `2_BasicChatbot_withMemory.ipynb`, compiled with an `InMemorySaver` checkpointer.
- Calls `load_dotenv()` so `OPENAI_API_KEY` (and any other env vars) are picked up from a `.env` file automatically.
- Exposes a single module-level `chatbot` object — this is what `5_Chatbot_Streamlit_Frontend.py` imports (`from Chatbot_Langgraph_Backend import chatbot`).
- Note: also imports `PromptTemplate`, `RunnableSequence`, and `PydanticOutputParser`, none of which are used by `chat_node` — likely left over from copying the file structure of `3_Persistence.ipynb`. Safe to trim if you want a leaner module.

### 4. `3_Persistence.ipynb` — Persistence Concepts
A standalone exploration of LangGraph's persistence layer (not the chatbot itself), covering:
- **Short-term memory** — retaining state within a thread
- **Fault tolerance** — recoverability via checkpoints
- **Human-in-the-loop** — pausing/inspecting state mid-graph
- **Time travel** — replaying/inspecting past states

Demonstrated using a two-node example graph (`joke_generator → explanation_generator`) with `InMemorySaver`, showing:
- Running independent conversations via separate `thread_id`s (`config1`, `config2`)
- Inspecting current state: `workflow.get_state(config)`
- Inspecting full state history: `workflow.get_state_history(config)`

### 5. `5_Chatbot_Streamlit_Frontend.py` — Web UI
- A Streamlit chat interface that:
  - Maintains `st.session_state['message_history']` to redraw prior messages on rerun
  - Sends user input to the LangGraph chatbot with a fixed `thread_id: 'thread_1'`
  - Displays both user and AI messages using `st.chat_message`
- Imports the compiled chatbot from a backend module:
  ```python
  from Chatbot_Langgraph_Backend import chatbot
  ```

## Setup

### Requirements
```bash
pip install langchain langchain-openai langgraph streamlit python-dotenv
```

### Environment Variables
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```
Both `3_Persistence.ipynb` and `Chatbot_Langgraph_Backend.py` already call `load_dotenv()`, so this is picked up automatically.

## Usage

### Run a notebook
Open any `.ipynb` file in Jupyter/VS Code and run cells top to bottom. `1_BasicChatbot.ipynb` and `2_BasicChatbot_withMemory.ipynb` end with an interactive CLI loop — type your message at the `TypeHere:` prompt, and `quit`/`exit`/`bye` to stop.

### Run the Streamlit app
```bash
streamlit run 5_Chatbot_Streamlit_Frontend.py
```
This opens a chat UI in your browser. All messages within a session share `thread_id: 'thread_1'`, so the bot remembers the conversation as you chat — but memory is in-RAM only (`InMemorySaver`) and resets when the app restarts.

## Known Limitations / Next Steps
- **In-memory only:** Chat history is lost on restart since `InMemorySaver` doesn't persist to disk. Consider swapping in a SQLite or Postgres checkpointer for durability.
- **Unused imports in `Chatbot_Langgraph_Backend.py`:** `PromptTemplate`, `RunnableSequence`, `PydanticOutputParser`, and `BaseModel`/`Field` are imported but never used — safe to remove for a cleaner backend module.
- **Single fixed thread in the UI:** The Streamlit app hardcodes `thread_id: 'thread_1'`, so all users/sessions share one conversation thread. Generating a unique `thread_id` per browser session would isolate conversations properly.
- **No streaming:** Responses are returned in full rather than streamed token-by-token.
- **`3_Persistence.ipynb` is exploratory,** not wired into the chatbot — its concepts (state history, time travel) could be added as chatbot features (e.g., "undo last message," "view conversation history").