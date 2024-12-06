import os

import streamlit as st
from PyPDF2 import PdfReader
# for local use
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)


# App
# Sessions
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
loading_placeholder = st.empty()
input_placeholder = st.empty()
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
            st.session_state.user_prompt = True
            user_prompt = st.text_input('Enter a question related to the provided documents:',
                                        disabled=False,
                                        placeholder="What's your question ?")
        except Exception as e:
            st.sidebar.write(f"⚠️ Error embedding documents: \n\n {e}")

    st.sidebar.write(st.session_state.vectorstore_ready)
    st.sidebar.write(st.session_state.ask)

    # Main interface
    if st.session_state.user_prompt and user_prompt:
        retriever = st.session_state.vectors.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(user_prompt)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = ChatPromptTemplate.from_template(
            '''
            You are a helpful question-answering assistant called DocuBot.
            Answer the questions based on the provided context only.
            Provide the most accurate response based on the question.
            Make the answer easy to understand.
            If there is no matching answers in the context say that The documents provided do not contain answers to this question.
            If you get asked who made you, say that you were made by Wassim Hamra.
            <context> 
            {context}
            </context>
            Question:{input}
            ''')
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'context': context, 'input': user_prompt})

        st.markdown(
            f"""
            <style>
            .box {{
                margin: 0em 0em 2em 0em;
                background: -webkit-linear-gradient(bottom left, red, orange);
                background: linear-gradient(to top right, red, orange);
                padding: 10px;
                border-radius: 8px;
                color: white;
                text-align: left;
                font-weight:bold;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="box"><p><strong>DocuBot:</strong> {response['answer']}</p></div>',
            unsafe_allow_html=True)

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
            background-color: #0C0C0C;
            text-align: center;
            padding-left:320px;
            font-size: 10px;
            color: white;
        }
    </style>
    <div class="footer">
        <p>Made with ❤️ by <a href='https://wassimhamra.com/'>Wassim Hamra</a></p>
    </div>
    """,
    unsafe_allow_html=True
)