# 📌 Full QA Pipeline with Pinecone + Perplexity + HuggingFace embeddings

import os
from dotenv import load_dotenv
import langchain
import pinecone

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_perplexity.chat_models import ChatPerplexity
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


# ==============================
# 1. Load Environment Variables
# ==============================
load_dotenv()


# ==============================
# 2. Read PDF Documents
# ==============================
def read_doc(directory):
    """Loads all PDFs from a given directory"""
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


# ==============================
# 3. Chunk Data
# ==============================
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """Splits documents into smaller chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


# ==============================
# 4. Setup Embeddings
# ==============================
def get_embeddings(model="huggingface"):
    """Choose embeddings: huggingface (default) or openai"""
    if model == "openai":
        return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ==============================
# 5. Setup Pinecone
# ==============================
def setup_pinecone(index_name="langchainqachatbot", dim=384):
    """Initializes Pinecone client and index"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dim,  # depends on embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return PineconeVectorStore


# ==============================
# 6. Create Vector Store
# ==============================
def build_vectorstore(documents, embeddings, index_name="langchainqachatbot"):
    """Builds Pinecone vectorstore from documents"""
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore


# ==============================
# 7. QA Chain with Perplexity
# ==============================
def setup_qa_chain():
    llm = ChatPerplexity(model="sonar-pro", temperature=0.5)

    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the given context:\n\n{context}\n\nQuestion: {input}"
    )

    chain = create_stuff_documents_chain(llm, prompt)
    return chain


# ==============================
# 8. Retrieve & Answer
# ==============================
def retrieve_query(vectorstore, query, k=2):
    return vectorstore.similarity_search(query, k=k)


def retrieve_answers(chain, vectorstore, query):
    doc_search = retrieve_query(vectorstore, query)
    response = chain.invoke({"context": doc_search, "input": query})
    return response


# ==============================
# 9. Run Full Pipeline
# ==============================
if __name__ == "__main__":
    # Step 1: Load and chunk documents
    raw_docs = read_doc("documents/")
    print(f"Loaded {len(raw_docs)} documents")

    chunked_docs = chunk_data(raw_docs)
    print(f"Split into {len(chunked_docs)} chunks")

    # Step 2: Setup embeddings
    embeddings = get_embeddings("huggingface")

    # Step 3: Setup Pinecone + vectorstore
    setup_pinecone(index_name="langchainqachatbot", dim=384)
    vectorstore = build_vectorstore(chunked_docs, embeddings)

    # Step 4: Setup QA chain
    chain = setup_qa_chain()

    # Step 5: Ask queries
    query = "How much the agriculture target will be increased by how many crore?"
    answer = retrieve_answers(chain, vectorstore, query)

    print("\n🔍 Query:", query)
    print("💡 Answer:", answer)
