# 🤖 DocuBot: PDF Question-Answering Chatbot using RAG

DocuBot is a PDF-based Question-Answering (QA) chatbot built using **Retrieval-Augmented Generation (RAG)**. This chatbot allows users to upload PDF documents and interact with them to retrieve answers to their questions based on the content of the documents. It uses **LangChain** and **HuggingFace** models to process the documents and provide relevant answers.
*  ***Web App link: [pdf-question-answering-chatbot-wassim-hamra.streamlit.app](https://pdf-question-answering-chatbot-wassim-hamra.streamlit.app/)***
* **⚠️ Wake the Streamlit Application if it's sleeping**
## Features

- **Upload multiple PDF files**: Users can upload multiple PDF files to interact with.
- **Document Embedding**: Once documents are uploaded, they are embedded into a vector space for efficient searching.
- **Question-Answering**: Ask questions related to the provided documents, and DocuBot will return the most relevant answers.
- **Document Similarity Search**: Explore documents related to the question for more context.

## Project Structure

### 1. **`Load PDFs`**: 
   - Takes the uploaded PDFs and extracts text from them.
   - Returns the documents in a format that can be processed by the model.

### 2. **`Vector Embedding`**:
   - Embeds the documents into vectors using **HuggingFace Embeddings** and stores them in a **FAISS** vector store.
### 2. **`LLM`**:
   - Used Groq to access the **gemma2-9b-it** model.
### 4. **`Interface`**:
   - User interacts with the chatbot via a simple text input field.
   - Once the documents are embedded, the user can ask questions and receive answers based on the context of the uploaded documents.
### 4. **`Deployment`**:
   - Used Streamlit Cloud for deployment
      *  ***Web App link: [pdf-question-answering-chatbot-wassim-hamra.streamlit.app](https://pdf-question-answering-chatbot-wassim-hamra.streamlit.app/)***
     * **⚠️ Wake the Streamlit Application if it's sleeping**

## Prerequisites

- **Python 3.10+**
- **Streamlit** for the user interface
- **LangChain** for document processing and retrieval
- **FAISS** for storing and querying document embeddings
- **HuggingFace** models for embeddings

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-repo/DocuBot
   cd DocuBot
2. **Set up a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
4. **Set up environment variables (Create a .env file and add the following keys):**

   ```bash
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=your_langchain_endpoint
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_PROJECT=your_langchain_project
   HF_TOKEN=your_huggingface_token
5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
