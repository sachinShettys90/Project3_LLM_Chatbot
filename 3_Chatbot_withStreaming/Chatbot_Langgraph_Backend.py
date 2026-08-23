from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, List, Annotated, Dict
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# Basemessage(includes AI , Human message)
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
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


checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


''' Lets implement the Streaming part in the code
*****sample code to implement streaming part in chatbot****

***we have to implement this in frontend code****

for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content='What is the receipe to make pasta')]}, config={'configurable': {'thread_id': 'thread-1'}}, stream_mode='messages'):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)

'''
