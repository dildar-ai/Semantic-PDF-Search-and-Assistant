import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI PDF Semantic Search", layout="wide")
st.title("📄 AI PDF Semantic Search & Q&A Assistant")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔑 Configuration")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API Key.")
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
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        st.success("✅ PDF processed successfully")

    # -----------------------------
    # User Question
    # -----------------------------
    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):
            docs = retriever.get_relevant_documents(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are an AI assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)

            st.subheader("🧠 Answer")
            st.write(response.content)

            with st.expander("📌 Retrieved Context"):
                st.write(context)
