Threading implementation in chat bot steps

-> add a sidebar with title + A Start Chat Button + A title named 'My Conversations'

-> generate dynamic thread id and add it to the session------------> Use python library UUID and define the fuction to generate threaid

-> Display the thread id in sidebar

********************************************************************************

-> add a New Chat button

-> On click of new chat open a new chat window
     * generate a new thread_id
     * save it in session
     * reset message history


def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['message_history']=[]

if st.sidebar.button('New Chat'):
    reset_chat()
    
********************************************************************************

-> create a list to store all thread_ids

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

*2 places we have to add threadid*
1.When we load the page
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
add_thread(st.session_state['thread_id'])

2.When we click on New Chat button also we have to add threadid
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    *add_thread(st.session_state['thread_id'])*
    st.session_state['message_history'] = []


-> Load all the thread ids in the sidebar

-> convert the side bar text to clickable buttons

********************************************************************************

-> on click of a particular thread id load that particular conversation