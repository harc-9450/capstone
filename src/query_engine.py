import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def load_qa_pipeline(persist_dir="chroma_db", collection_name="excel_docs", llm_model="gemini-2.0-flash"):
    """Loads a RetrievalQA chain with Gemini LLM and Chroma retriever."""
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
