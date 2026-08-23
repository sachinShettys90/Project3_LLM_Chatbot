import streamlit as st
from Chatbot_Langgraph_Backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread_1'}}

# st.session_state--->streamlit dictionary , which will store the dictionary for that session state
# instead of using this message_history = [] ,use the st.session_state dictionary
# message_history = []

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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

    with st.chat_message("AI"):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]}, config={'configurable': {'thread_id': 'thread-1'}}, stream_mode='messages')
        )
    st.session_state['message_history'].append(
        {'role': 'AI', 'content': ai_message})


# here from the chatbot.stream , we will get the message_chunk and metadata , where we are getting the message_chunk.content and passing it to st.write_stream and pass it to ai_message
