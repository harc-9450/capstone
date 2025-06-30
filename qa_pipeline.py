import os
import chromadb
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


load_dotenv()  # Loads from .env file automatically


def load_qa_pipeline(embedding_model_name, db_folder, k=5, llm_model="gemini-2.0-flash"):

    # Use LangChain-compatible HuggingFaceEmbeddings wrapper
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load vector DB
    vectordb = Chroma(
        collection_name="movie_summaries",
        embedding_function=embedding,
        persist_directory=db_folder
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))  # You can swap this out later

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def answer_question(query, embedding_model_name, db_path,  llm_model="gemini-2.0-flash"):
    pipeline = load_qa_pipeline(embedding_model_name, db_path, llm_model=llm_model)
    return pipeline.invoke({"query": query})
