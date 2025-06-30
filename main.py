import os
from dotenv import load_dotenv
from src.data_loader import load_excel_as_documents
from src.chunker import convert_to_documents
from src.embed_store import embed_and_store
from src.query_engine import load_qa_pipeline
import pandas as pd

load_dotenv()

# Step 1: Load Excel and flatten rows
excel_path = "data/Canada.xlsx"
flat_docs = load_excel_as_documents(excel_path)

# Step 2: Convert to LangChain Documents
documents = convert_to_documents(flat_docs)

# Step 3: Check if ChromaDB already exists
db_path = "chroma_db"
if os.path.exists(db_path) and os.path.isdir(db_path):
    print("📦 Loading existing ChromaDB...")
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding,
        collection_name="excel_docs"
    )
else:
    print("🔄 Embedding and storing documents...")
    vectordb = embed_and_store(documents)

# Step 4: Load QA system
print("✅ Loading QA system...")
qa = load_qa_pipeline()

# Step 5: Ask question
query = input("\\n❓ Enter your question about the Excel data: ")
response = qa.invoke(query)

print("\\n🤖 Gemini Answer:")
print(response["result"] if isinstance(response, dict) else response)

# Step 6: Show matched rows
print("\\n📄 Top Matching Chunks:")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)

for d in docs:
    print(f"📝 Sheet: {d.metadata['sheet']}, Row: {d.metadata['row']}")
    try:
        row_dict = dict(item.split(": ", 1) for item in d.page_content.split(" | "))
        row_df = pd.DataFrame([row_dict])
        print(row_df.to_markdown(index=False))
    except ImportError:
        print("📎 Install 'tabulate' to enable markdown table output.")
        print(pd.DataFrame([row_dict]))
    print()