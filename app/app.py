import os
import pickle

import pandas as pd
import requests
import streamlit as st

TMDB_API_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500{}"


def get_tmdb_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY")


def fetch_poster(movie_id, api_key):
    if not api_key:
        return None

    response = requests.get(
        TMDB_API_URL.format(movie_id),
        params={"api_key": api_key, "language": "en-US"},
        timeout=10,
    )
    response.raise_for_status()

    poster_path = response.json().get("poster_path")
    return TMDB_IMAGE_URL.format(poster_path) if poster_path else None


@st.cache_resource
def load_model():
    with open("movie_dict.pkl", "rb") as file:
        movies = pd.DataFrame(pickle.load(file))

    with open("similarity_1.pkl", "rb") as file:
        similarity = pickle.load(file)

    return movies, similarity


def recommend(movie_title, movies, similarity):
    matches = movies.index[movies["title"] == movie_title].tolist()

    if not matches:
        return []

    distances = similarity[matches[0]]
    top_movies = sorted(
        enumerate(distances),
        key=lambda item: item[1],
        reverse=True,
    )[1:6]

    return [
        {
            "title": movies.iloc[index]["title"],
            "movie_id": movies.iloc[index]["movie_id"],
            "score": score,
        }
        for index, score in top_movies
    ]


st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
)

st.title("Movie Recommendation System")
st.write(
    "A content-based recommender built from movie metadata "
    "using CountVectorizer and cosine similarity."
)

api_key = get_tmdb_api_key()

try:
    movies, similarity = load_model()
except FileNotFoundError as error:
    st.error(f"Required model artifact is missing: {error.filename}")
    st.info(
        "Run the model-generation notebook to create the required artifacts "
        "before starting the Streamlit application."
    )
    st.stop()

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].dropna().sort_values().unique(),
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, movies, similarity)

    if not recommendations:
        st.warning("No recommendations were found.")
        st.stop()

    columns = st.columns(len(recommendations))

    for column, recommendation in zip(columns, recommendations):
        with column:
            st.subheader(recommendation["title"])

            if api_key:
                try:
                    poster = fetch_poster(recommendation["movie_id"], api_key)
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.caption("Poster unavailable.")
                except requests.RequestException:
                    st.caption("Poster unavailable.")
            else:
                st.caption("Set TMDB_API_KEY to display posters.")

            st.caption(f"Similarity: {recommendation['score']:.3f}")
