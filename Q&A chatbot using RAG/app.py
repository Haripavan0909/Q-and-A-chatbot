# app.py
import streamlit as st
from rag_pipeline import read_doc, chunk_data, get_embeddings, setup_pinecone, build_vectorstore, setup_qa_chain, retrieve_answers

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="RAG QA Bot", page_icon="🤖", layout="wide")

st.title("📚 RAG-powered QA Bot")
st.write("Ask questions from your PDF documents using Pinecone + HuggingFace + Perplexity")

# Sidebar for configuration
st.sidebar.header("⚙️ Settings")
index_name = st.sidebar.text_input("Pinecone Index Name", "langchainqachatbot")
embedding_type = st.sidebar.selectbox("Embedding Model", ["huggingface", "openai"])
top_k = st.sidebar.slider("Top K results", 1, 5, 2)

# File uploader
uploaded_files = st.file_uploader("📂 Upload Documents", type=None, accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📑 Processing documents..."):
        # Save uploaded files temporarily
        import os
        os.makedirs("uploaded_docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join("uploaded_docs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Run pipeline
        raw_docs = read_doc("uploaded_docs/")
        st.success(f"Loaded {len(raw_docs)} documents")

        chunked_docs = chunk_data(raw_docs)
        st.success(f"Split into {len(chunked_docs)} chunks")

        embeddings = get_embeddings(embedding_type)
        setup_pinecone(index_name=index_name, dim=384)
        vectorstore = build_vectorstore(chunked_docs, embeddings, index_name=index_name)
        chain = setup_qa_chain()

        st.success("✅ Vectorstore and QA chain ready!")

        # Ask Questions
        query = st.text_input("🔍 Ask your question:")
        if query:
            with st.spinner("🤔 Thinking..."):
                answer = retrieve_answers(chain, vectorstore, query)
                st.markdown("### 💡 Answer")
                st.write(answer)
