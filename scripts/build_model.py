import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"


def convert(value):
    items = ast.literal_eval(value)
    return [item["name"] for item in items[:3]]


def fetch_director(value):
    for item in ast.literal_eval(value):
        if item.get("job") == "Director":
            return [item["name"]]
    return []


def build_model():
    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    movies = movies.merge(credits, on="title")

    for column in ["genres", "keywords", "cast"]:
        movies[column] = movies[column].apply(convert)

    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].fillna("")
    movies["tags"] = (
        movies["overview"].str.split()
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    model_df = movies[["movie_id", "title", "tags"]].copy()
    model_df["tags"] = model_df["tags"].apply(
        lambda values: " ".join(values).lower()
    )

    stemmer = PorterStemmer()
    model_df["tags"] = model_df["tags"].apply(
        lambda text: " ".join(stemmer.stem(word) for word in text.split())
    )

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(model_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    with open(ROOT / "movie_dict.pkl", "wb") as file:
        pickle.dump(model_df.to_dict(), file)

    with open(ROOT / "similarity_1.pkl", "wb") as file:
        pickle.dump(similarity, file)

    print(f"Created model for {len(model_df)} movies.")


if __name__ == "__main__":
    build_model()
