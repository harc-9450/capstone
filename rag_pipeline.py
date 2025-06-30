import os
import fitz  
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CHROMA_DB_DIR = "embeddings/chroma_db"
DOCS_DIR = "docs/"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# === Step 1: Extract and segment the PDF ===
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()

    # Simple segmentation by headings — can improve with regex
    sections = {}
    current_section = "Unknown"
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith(("abstract", "introduction", "method", "methodology", "experiment", "results", "conclusion", "references")):
            current_section = line
            sections[current_section] = ""
        elif current_section in sections:
            sections[current_section] += line + " "
    return sections

# === Step 2: Chunk the section text ===
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === Step 3: Embed and store in ChromaDB ===
def embed_sections(sections, doc_name="paper"):
    embedder = SentenceTransformer(EMBED_MODEL)
    client = PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection("research_docs")

    all_chunks = 0
    for section_name, section_text in sections.items():
        chunks = chunk_text(section_text)
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            metadata = {
                "section": section_name,
                "source": doc_name
            }
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"{doc_name}-{section_name}-{i}"]
            )
            all_chunks += 1
    print(f"✅ Stored {all_chunks} chunks in ChromaDB.")


# === Step 3: Rerank the chunks based on similarity ===
def rerank_chunks(query, chunks, embedder):
    query_embedding = embedder.encode([query])
    chunk_embeddings = embedder.encode([c["chunk"] for c in chunks])

    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Pair each chunk with its score
    reranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],  # sort by score descending
        reverse=True
    )
    
    return [item[0] for item in reranked]

# === Add Retrieval Functionality ===
def search_similar_chunks(query, top_k=5, filter_docs=None):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embedder.encode(query).tolist()

    client = PersistentClient(path="embeddings/chroma_db")
    collection = client.get_collection("research_docs")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,
        include=["documents", "metadatas"]
    )

    if not results["documents"] or not results["metadatas"]:
        return []

    # Filter by selected documents (if any)
    combined = [
        {
            "chunk": doc,
            "source": meta.get("source", "Unknown"),
            "section": meta.get("section", "N/A")
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        if (not filter_docs or meta.get("source", "") in filter_docs)
    ]
    
    reranked = rerank_chunks(query, combined, embedder)
    return reranked[:top_k]

# === Step 4: Run the full pipeline ===
def run_pipeline():
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCS_DIR, filename)
            print(f"📄 Processing {filename}...")
            sections = extract_sections(path)
            embed_sections(sections, doc_name=filename.replace(".pdf", ""))