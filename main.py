import os
from dotenv import load_dotenv
import sys
sys.path.append('src')
from loader import load_txt_documents
from embed_store import embed_and_store
from query_engine import load_qa_pipeline
import pandas as pd

load_dotenv()

# Step 1: Load .txt files from data/
folder_path = "data"
documents = load_txt_documents(folder_path)

# Step 2: Embed & store, or load existing ChromaDB
db_path = "chroma_db_txt"
if os.path.exists(db_path) and os.path.isdir(db_path):
    print("📦 Reusing existing Chroma vector DB...")
    qa = load_qa_pipeline()
else:
    print("🔄 Embedding and creating Chroma vector DB...")
    vectordb = embed_and_store(documents)
    qa = load_qa_pipeline()

# Step 3: Ask a question
query = input("\\n❓ Enter your question about the .txt documents: ")
response = qa.invoke(query)

print("\\n🤖 Gemini Answer:")
print(response["result"] if isinstance(response, dict) else response)

# Step 4: Show matching source files
print("\\n📄 Top Matching Chunks:")
docs = qa.retriever.get_relevant_documents(query)

for d in docs:
    print(f"📝 Source File: {d.metadata.get('source_file', 'unknown')}")
    print("✂️ Snippet:")
    print(d.page_content[:300] + "...")
    print()