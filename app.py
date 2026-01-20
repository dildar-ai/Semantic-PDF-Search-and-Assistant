import streamlit as st
import tempfile
from pypdf import PdfReader
from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# -----------------------------
# Simple Text Splitter (OWN)
# -----------------------------
def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI PDF Semantic Search", layout="wide")
st.title("📄 AI PDF Semantic Search & Q&A Assistant")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🔑 Configuration")
groq_api_key = "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"

if not groq_api_key:
    st.warning("Please enter your Groq API Key.")
    st.stop()


# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):

        reader = PdfReader(uploaded_file)
        full_text = ""

        for page in reader.pages:
            full_text += page.extract_text()

        chunks = split_text(full_text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_texts(chunks, embeddings)

        st.success("✅ PDF processed successfully!")


    # -----------------------------
    # User Question
    # -----------------------------
    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):

            docs = vectorstore.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are an AI assistant. Answer the question ONLY using the context below.

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
