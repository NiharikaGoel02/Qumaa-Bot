import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# PAGE CONFIG
st.set_page_config(page_title="Kumaa Chatbot", page_icon="🤖", layout="wide")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# TITLE
st.title("🤖 Welcome to Kumaa — Your AI Assistant")

# FILE UPLOADER
uploaded_file = st.file_uploader("📄 Upload a PDF to chat with:", type=["pdf"])

if uploaded_file is not None:
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp_uploaded_file.pdf")  
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Embed chunks
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    llm = Ollama(model="gemma:2b")

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        input_key="question",    
        output_key="answer" 
    )

    # Build ConversationalRetrievalChain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
    output_key="answer" 
)

# CHAT
input_text = st.text_input("💬 Ask your question:")

if input_text:
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    with st.chat_message("assistant"):
        try:
            if st.session_state.qa_chain is not None:
                # Chat over PDF
                result = st.session_state.qa_chain.invoke({"question": input_text})
                st.markdown(result["answer"])
            else:
                # Normal LLM chat
                llm = Ollama(model="gemma:2b")
                simple_answer = llm.invoke(input_text)
                st.markdown(simple_answer)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
