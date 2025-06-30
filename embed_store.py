from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def embed_and_store(summary_df, model_name="all-MiniLM-L6-v2", db_path=None):

    model = SentenceTransformer(model_name)

    # Default to model-named folder if db_path not given
    if db_path is None:
        db_path = f"embeddings/chroma_db_{model_name.replace('/', '_')}"

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("movie_summaries")

    print(f"🚀 Embedding with model: {model_name} | DB Path: {db_path}")
    
    for _, row in summary_df.iterrows():
        doc_id = str(row["MovieID"])
        text = row["Summary"]
        embedding = model.encode(text).tolist()

        metadata = {
            "title": row["Title"],
            "genres": row["Summary"].split(" is a ")[1].split(" movie")[0]
        }

        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata]
        )

    print("✅ Done embedding and storing.")
