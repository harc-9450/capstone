import pandas as pd

def load_and_merge_movies_data(movies_path, ratings_path, users_path):
    # File paths
    # movies_path = "./data/movies.dat"
    # ratings_path = "./data/ratings.dat"
    # users_path = "./data/users.dat"
    try:
        # Load files with :: separator and proper encoding
        movies = pd.read_csv(movies_path, sep="::", engine="python", encoding="ISO-8859-1",
                             names=["MovieID", "Title", "Genres"])

        ratings = pd.read_csv(ratings_path, sep="::", engine="python", encoding="ISO-8859-1",
                              names=["UserID", "MovieID", "Rating", "Timestamp"])

        users = pd.read_csv(users_path, sep="::", engine="python", encoding="ISO-8859-1",
                            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])

        # Merge ratings + users → then join with movies
        ratings_users = pd.merge(ratings, users, on="UserID", how="inner")
        full_data = pd.merge(ratings_users, movies, on="MovieID", how="inner")

        print(f"✅ Loaded and merged {len(full_data)} records successfully.")
        return full_data

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()
