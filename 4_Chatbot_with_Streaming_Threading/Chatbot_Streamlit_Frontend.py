import streamlit as st
from Chatbot_Langgraph_Backend import chatbot
from langchain_core.messages import HumanMessage

import uuid   # to generate the random thread_id


# st.session_state--->streamlit dictionary , which will store the dictionary for that session state
# instead of using this message_history = [] ,use the st.session_state dictionary
# message_history = []

# *************************************************Utility functions********************************


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

# *********************************************************************************************


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:  # if the thread id is not there generate it
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []


# *************************************************Sidebar UI********************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'AI'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.session_state['thread_id'] = thread_id
        st.rerun()


# ******************************************************************************************

# loading the conversation hitory
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type here")

if user_input:
    add_thread(st.session_state['thread_id'])
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
