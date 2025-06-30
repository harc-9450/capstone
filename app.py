import os
import pandas as pd

from data_loader import load_and_merge_movies_data
from embed_store import embed_and_store
from qa_pipeline import answer_question
from summary_generator import generate_movie_summaries

#File paths
movies_path = "./data/movies.dat"
ratings_path = "./data/ratings.dat"
users_path = "./data/users.dat"

models_to_try = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L12-v2"
]

# Load and prepare summaries once
df = load_and_merge_movies_data("data/movies.dat", "data/ratings.dat", "data/users.dat")
summary_df = generate_movie_summaries(df)


# print(summary_df.head())

# embed and store for each model
all_exist = all(
    os.path.exists(f"embeddings/chroma_db_{model_name.replace('/', '_')}")
    for model_name in models_to_try
)
if not all_exist:
    print("🔁 Embedding and storing summaries for each model...")
    for model_name in models_to_try:
        print(f"\n🔁 Running with model: {model_name}")
        embed_and_store(summary_df, model_name=model_name)



#Load the pipeline and run a query

query = "Which sci-fi movie is highly rated by young adults?"
model_name = "all-MiniLM-L6-v2"
db_path = f"embeddings/chroma_db_{model_name.replace('/', '_')}"

response = answer_question(query, model_name, db_path)
print("💬 Answer:", response)

