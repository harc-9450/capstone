import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

PDF_PATH = "docs/COBECPolicy_v1.pdf"
CHROMA_DB_DIR = "embeddings/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# STEP 1: Load PDF and extract text
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# STEP 2: Split text into overlapping chunks
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# step 3: Embed and store in ChromaDB
def embed_and_store(chunks, version="v1", access_level="All"):
    #initialize sentence transformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    #initialize ChromaDB client
    chroma_client = PersistentClient(path=CHROMA_DB_DIR)

    #create collection ( or get if exists)
    collection = chroma_client.get_or_create_collection("cobec_docs")

    #embed and insert chunks
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        metadata = {
            "version": version,
            "access_level": access_level
        }
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{version}-chunk-{i}"],
            metadatas=[metadata]
        )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB with metadata.")

# Run the pipeline

def run_pipeline():
    pdf_dir = "docs/"
    print("🔍 Scanning for PDF files...")

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf") and "COBECPolicy" in filename:
            # Parse version and access from filename (e.g., COBECPolicy_v1_HR.pdf)
            parts = filename.replace(".pdf", "").split("_")
            if len(parts) >= 3:
                version = parts[1]
                access = parts[2]
            else:
                version = "v1"
                access = "All"

            print(f"📄 Processing {filename} | version={version}, access={access}")
            text = extract_pdf_text(os.path.join(pdf_dir, filename))
            chunks = chunk_text(text)
            embed_and_store(chunks, version=version, access_level=access)


# STEP 4: Retrieve relevant chunks based on query

def search_similar_chunks(query, top_k=3, version="v1", access_level="All"):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode(query).tolist()

    chroma_client = PersistentClient(path=CHROMA_DB_DIR)
    collection = chroma_client.get_collection("cobec_docs")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        include=["documents", "metadatas"]
    )

    # ✅ Check if results are empty
    if not results["documents"] or not results["documents"][0]:
        return []

    # ✅ Now safely filter
    filtered = [
        doc for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        if meta.get("version") == version and (
            meta.get("access_level") == "All" or meta.get("access_level") == access_level
        )
    ]
    return filtered[:top_k]