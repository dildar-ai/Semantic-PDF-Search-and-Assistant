import streamlit as st
import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="AI PDF Semantic Search", layout="wide")
st.title("📄 AI PDF Semantic Search & Q&A Assistant")

# -----------------------------
# Sidebar - API Key
# -----------------------------
st.sidebar.title("🔑 Configuration")
groq_api_key = "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"

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        st.success("✅ PDF processed successfully!")

    # -----------------------------
    # User Question
    # -----------------------------
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)

            st.subheader("🧠 Answer")
            st.write(result["result"])

            with st.expander("📌 Source Chunks"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
                    st.markdown("---")
