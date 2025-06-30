import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def embed_and_store(docs, persist_dir="chroma_db", collection_name="excel_docs"):
    """Embeds documents using Gemini and stores in ChromaDB."""
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb
