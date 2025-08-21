# Q-and-A-chatbot
This project is an end-to-end Question & Answer Chatbot built using Large Language Models (LLMs) + Retrieval Augmented Generation (RAG).
It allows users to upload documents (PDF, Word, Text, etc.) and then ask natural language questions. The chatbot retrieves relevant chunks from the documents and provides accurate, context-driven answers.

This project combines the power of Retrieval-Augmented Generation (RAG) with modern tools and frameworks:

- **LLM Models:** Hugging Face, OpenAI, Perplexity (for embeddings & QA)

- **Vector Database:** Pinecone (for semantic search & retrieval)

- **Frameworks:** LangChain (for pipeline integration), Streamlit (for frontend)

- **Document Handling:** PyPDFLoader, Docx2txt, TextLoader (for multiple file types)

✅ Key Features:

- Upload multiple types of documents (PDF, DOCX, TXT, etc.)
- Automatic document preprocessing & text chunking
- Vector Embeddings generation with Hugging Face / OpenAI
- Store & retrieve embeddings using Pinecone Vector DB
- RAG pipeline for accurate question answering
- Interactive Streamlit UI for easy use

**1️⃣ Create a Virtual Environment**

>> python -m venv venv
>> 
>> venv\Scripts\activate 

**2️⃣ Install Dependencies**

>> pip install -r requirements.txt

**3️⃣ Setup Environment Variables**

>> OPENAI_API_KEY=your_openai_api_key
>> 
>> PINECONE_API_KEY=your_pinecone_api_key
>>
>> PINECONE_ENV=your_pinecone_environment

**4️⃣ Commands used in Project**

>> pip install streamlit                         #for frontend
>>
>> pip install pinecone-client                   #for vector database integration
>>
>> pip install openai                            #for embeddings & LLM calls
>>
>> pip install python-dotenv                     #to load .env API keys securely
>>
>> pip install langchain                         #to build the Retrieval Augmented Generation pipeline
>>
>> pip install langchain-community                   
>>
>> pip install langchain-pinecone
>>
>> pip install langchain-openai
>>
>> pip install langchain-huggingface


# **📊 Pros & Cons**

**✅ Pros**

- Works with any type of document (multi-format support)

- Provides accurate, context-based answers

- Scalable with Pinecone vector DB

- Easy to use via Streamlit UI

**⚠️ Cons**

- Depends on API costs (OpenAI, Pinecone, etc.)

- Requires good chunking strategy for large docs

- Internet connection required for APIs


# **🚀 How to Run**

Once you’ve completed the setup, run the following commands:

**1️⃣ Run the RAG pipeline directly**

>> python rag_pipeline.py

**2️⃣ Launch the Streamlit Web App**

>> streamlit run app.py
