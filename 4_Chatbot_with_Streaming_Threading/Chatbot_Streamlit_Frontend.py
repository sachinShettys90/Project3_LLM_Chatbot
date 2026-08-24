import streamlit as st
from Chatbot_Langgraph_Backend import chatbot
from langchain_core.messages import HumanMessage

import uuid   # to generate the random thread_id


# st.session_state--->streamlit dictionary , which will store the dictionary for that session state
# instead of using this message_history = [] ,use the st.session_state dictionary
# message_history = []

# *************************************************Utility functions********************************


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

# *********************************************************************************************


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:  # if the thread id is not there generate it
    st.session_state['thread_id'] = generate_thread_id()


# *************************************************Sidebar UI********************************
st.sidebar.title("LangGraph Chatbot")
st.sidebar.button('New Chat')
st.sidebar.header("My Conversations")

st.sidebar.text(st.session_state['thread_id'])

# ******************************************************************************************

# loading the conversation hitory
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type here")

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message("AI"):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages')
        )

        st.session_state['message_history'].append(
            {'role': 'AI', 'content': ai_message})

        # here from the chatbot.stream , we will get the message_chunk and metadata , where we are getting the message_chunk.content and passing it to st.write_stream and pass it to ai_message
