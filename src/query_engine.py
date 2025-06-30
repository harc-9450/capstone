import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def load_qa_pipeline(persist_dir="chroma_db_txt", collection_name="text_docs", llm_model="gemini-2.0-flash"):

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)