import base64
import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Loading environ variables - for local use
# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# for deployment
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = str(st.secrets["LANGCHAIN_TRACING_V2"])
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

llm = ChatGroq(model_name='gemma2-9b-it')


# Functions
def load_pdfs(uploaded_files):
    docs = []
    for pdf_file in uploaded_files:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += '\n\n' + page.extract_text()
        doc = Document(page_content=text)
        docs.append(doc)
    return docs


def create_vector_embedding(docs):
    if 'vectors' not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=250)
        st.session_state.final_docs = st.session_state.splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)

def get_session_history(session:str)-> BaseChatMessageHistory:
    if st.session_state.id not in st.session_state.store:
        st.session_state.store[st.session_state.id] = ChatMessageHistory()
    return st.session_state.store[st.session_state.id]

def message_func(content, is_user=False):
    bubble_color = "linear-gradient(to bottom left, red, orange)" if is_user else "linear-gradient(to top right, red, orange)"
    alignment = "right" if is_user else "left"
    avatar_url = './images/avatar_user.png' if is_user else './images/avatar_chatbot.png'
    with open(avatar_url, "rb") as img_file:
        avatar_image = img_file.read()
    st.markdown(
    f"""
    <div style="text-align: {alignment}; margin: 5px 0; display: flex; align-items: center; flex-direction: {('row-reverse' if alignment == 'right' else 'row')};">
    <img src="data:image/png;base64,{base64.b64encode(avatar_image).decode()}" style="width: 40px; height: 40px; border-radius: 50%; margin-left: 10px;margin-right: 10px;" alt="Avatar">
            <div style="
            display: inline-block;
            padding: 13px 13px 0px 13px;
            border-radius: 20px;
            background: {bubble_color};
            color: white;
            max-width: 70%;
            word-wrap: break-word;
            font-family: optima, sans-serif;
            font-size: 0.8em;">
            <p><strong>{content}</strong></p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# App
# Sessions
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'nb_documents' not in st.session_state:
    st.session_state.nb_documents = False
if 'embedded_documents' not in st.session_state:
    st.session_state.embedded_documents = False
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = ''
if 'ask' not in st.session_state:
    st.session_state.ask = ''
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'id' not in st.session_state:
    st.session_state.id = 'default-session'
user_prompt = False

gradient_text_html = """
    <style>
    .gradient-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, red, orange);
        background: linear-gradient(to right, red, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        font-size: 3em;
        text-align: center;
        width: 100%;
        display: block;
    }
    </style>
    <div class="gradient-text">DocuBot🤖</div><br>
    <h3 class="gradient-text" style="font-weight: light; font-size: 2em;">A PDF Question-Answering Chatbot using RAG</h3>
    """

st.markdown(gradient_text_html, unsafe_allow_html=True)
st.subheader("")
# Sidebar
st.sidebar.subheader('DocuBot 🤖')
uploaded_files = st.sidebar.file_uploader('Upload your PDF files', type="pdf", accept_multiple_files=True)
def reset():
    st.session_state.chat_history = []

st.session_state.id = st.text_input('Session ID', value=st.session_state.id, disabled=False, on_change=reset)

gradient_text_html2 = """
    <style>
    .gradient-text_history {
        font-weight: light;
        margin-top: 30px;
        background: -webkit-linear-gradient(left, red, orange);
        background: linear-gradient(to right, red, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
        text-align: center;
        font-size: 2em;
        width: 100%;
        display: block;
    }
</style>
<h6 class="gradient-text_history">History</h6>

    """
if len(st.session_state.chat_history) > 1:
    st.markdown(gradient_text_html2, unsafe_allow_html=True)
loading_placeholder = st.empty()
session_placeholder = st.empty()
input_placeholder = st.empty()

for message in st.session_state.chat_history[1:]:
    message_func(message['content'], is_user=message['role'] == 'user')
if not st.session_state.user_prompt:
    input_placeholder.text_input('Enter a question related to the provided documents:',
                                 disabled=True, placeholder='❌ Please provide the documents and embed them first')

if uploaded_files:
    docs = load_pdfs(uploaded_files)
    st.session_state.nb_documents = f"Loaded {len(docs)} documents successfully ☑️"
    st.sidebar.write(st.session_state.nb_documents)
    st.session_state.embedded_documents = "Now, Embed your documents ⬇️"
    st.sidebar.write(st.session_state.embedded_documents)

    if st.sidebar.button('Embed Documents') or st.session_state.user_prompt:
        loading_placeholder.title(f"⏳ Just a moment…")
        try:
            create_vector_embedding(docs)
            st.session_state.vectorstore_ready = "Vectorstore ready ☑️"
            st.session_state.ask = "You can ask ️questions now ! 😊"
            loading_placeholder.empty()
            input_placeholder.empty()
            session_placeholder.empty()
            st.session_state.user_prompt = True
            user_prompt = st.text_input('Enter a question related to the provided documents:',
                                        disabled=False,
                                        placeholder="What's your question ?")
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        except Exception as e:
            st.sidebar.write(f"⚠️ Error embedding documents: \n\n {e}")

    st.sidebar.write(st.session_state.vectorstore_ready)
    st.sidebar.write(st.session_state.ask)

    # Main interface
    if st.session_state.user_prompt and user_prompt:
        retriever = st.session_state.vectors.as_retriever()
        retrieved_docs = retriever.invoke(user_prompt)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        session_history = get_session_history(st.session_state.id)
        #Prompts
        contextualize_q_system_prompt = (
            'Given a chat history and the latest user questions'
            'which might reference context in the chat history,'
            'formulate a standalone question which can be understood'
            'without the chat history. Do not answer the question,'
            'just reformulate it if needed and otherwise return it as it is.'
        )
        textualize_q_prompt = ChatPromptTemplate.from_messages(
           [
               ('system',contextualize_q_system_prompt),
               MessagesPlaceholder('chat_history',),
               ('human','{input}'),
           ])
        system_prompt = (
            'This is a message for you only, do not mention it when you answer the questions:'
            'You are a helpful question-answering assistant called DocuBot.'
            'Answer the questions based on the provided context only.'
            'Provide the most accurate response based on the question.'
            'Make the answer easy to understand.'
            'If there is no matching answers in the context say that The documents provided do not contain answers to this question.'
            'If you get asked who made you, say that you were made by Wassim Hamra.'
            '<context>'
            '{context}'
            '</context>'
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                MessagesPlaceholder('chat_history', ),
                ('human', '{input}'),
            ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, textualize_q_prompt)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer',
        )
        response = conversational_rag_chain.invoke({'context': context, 'input': user_prompt},
                                                   config = {'configurable': {"session_id": st.session_state.id}})
        st.session_state.chat_history.append({"role": "docubot", "content": response['answer']})
        message_func(user_prompt, is_user=True)
        message_func(response['answer'], is_user=False)
        with st.expander('Document Similarity Search'):
            for i, doc in enumerate(retrieved_docs):
                st.write(doc.page_content)
                st.write('-----------------------------')
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 5px;
            background: rgba(241, 241, 241, 0.7);
            text-align: center;
            padding-left:320px;
            font-size: 10px;
            color: black;
        }
    </style>
    <div class="footer">
        <p>Made with 🧡 by <a href='https://wassimhamra.com/'>Wassim Hamra</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
