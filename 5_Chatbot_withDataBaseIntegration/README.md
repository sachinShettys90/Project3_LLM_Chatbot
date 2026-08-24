User this command to install checkpoint sqlite
pip install langgraph-checkpoint-sqlite



Chatbot_Langgraph_Backend.py changes
# --------------Install Sqlitesaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3  # -------------> import this to define the database

# ---------------to save the database in local
conn=sqlite3.connect(database='chatbot.db', check_same_thread=False)   #this will build connection, pass it to conn of SqliteSaver
checkpointer = SqliteSaver(conn=conn)

# -------------to get all the threadid , define the function

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])


Chatbot_Streamlit_Frontend.py changes

from Chatbot_Langgraph_Backend import chatbot, retrieve_all_threads
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
