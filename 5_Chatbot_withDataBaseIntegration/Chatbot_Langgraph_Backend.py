from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, List, Annotated, Dict

# --------------Install Sqlitesaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3  # -------------> import this to define the database

# Basemessage(includes AI , Human message)
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = model.invoke(messages)
    # response store state
    return {'messages': [response]}


# to save the database in local
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


''' Lets implement the Streaming part in the code
*****sample code to implement streaming part in chatbot****

***we have to implement this in frontend code****

for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content='What is the receipe to make pasta')]}, config={'configurable': {'thread_id': 'thread-1'}}, stream_mode='messages'):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)

'''
