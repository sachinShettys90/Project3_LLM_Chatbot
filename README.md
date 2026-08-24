# LangGraph Chatbot

A persistent, streaming chatbot built with **LangGraph**, **LangChain**, and **Streamlit**, backed by SQLite for conversation history.

## Features

- 🔄 **Streaming responses** — tokens appear live as the model generates them
- 💾 **Persistent conversations** — every thread is checkpointed to SQLite via `langgraph`'s `SqliteSaver`, so chats survive app restarts
- 🗂️ **Multi-conversation sidebar** — switch between past chats, each labeled with an auto-generated title (from the first message) instead of a raw UUID
- 🗑️ **Delete conversations** — remove a thread and its history from the database
- ⚙️ **Configurable via environment variables** — swap models, temperature, system prompt, or DB path without touching code
- 🛡️ **Error handling** — a failed API call surfaces a readable message in the UI instead of crashing the app

## Architecture

```
Chatbot_Langgraph_Backend.py   # LangGraph StateGraph: single chat_node, SQLite checkpointer
Chatbot_Streamlit_Frontend.py  # Streamlit UI: sidebar thread list, chat window, streaming
```

The backend exposes a compiled `chatbot` graph plus small helper functions
(`retrieve_all_threads`, `load_conversation`, `get_thread_title`, `delete_thread`)
that the frontend calls — keeping UI and orchestration logic cleanly separated.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
streamlit run Chatbot_Streamlit_Frontend.py
```

## Possible next steps

- Swap `ChatOpenAI` for any other LangChain chat model to make the provider pluggable
- Add authentication so `thread_id`s are scoped per user
- Add a "regenerate response" / "edit message" action
- Track token usage per thread for cost visibility
