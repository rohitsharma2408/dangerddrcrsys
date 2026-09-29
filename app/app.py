import os
import pickle

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
)

TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


@st.cache_resource
def load_artifacts():
    with open("movie_dict.pkl", "rb") as file:
        movie_dict = pickle.load(file)

    with open("similarity_1.pkl", "rb") as file:
        similarity = pickle.load(file)

    return pd.DataFrame(movie_dict), similarity


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int):
    api_key = os.getenv("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY")

    if not api_key:
        return None

    response = requests.get(
        f"{TMDB_BASE_URL}/{movie_id}",
        params={"api_key": api_key, "language": "en-US"},
        timeout=10,
    )
    response.raise_for_status()

    poster_path = response.json().get("poster_path")
    return f"{TMDB_IMAGE_URL}{poster_path}" if poster_path else None


def recommend(movie_title: str, movies: pd.DataFrame, similarity):
    matches = movies.index[movies["title"] == movie_title]

    if len(matches) == 0:
        return []

    movie_index = matches[0]
    distances = similarity[movie_index]

    ranked = sorted(
        enumerate(distances),
        key=lambda item: item[1],
        reverse=True,
    )

    recommendations = []

    for index, score in ranked:
        if index == movie_index:
            continue

        row = movies.iloc[index]

        recommendations.append(
            {
                "title": row["title"],
                "movie_id": int(row["movie_id"]),
                "score": float(score),
            }
        )

        if len(recommendations) == 5:
            break

    return recommendations


st.title("🎬 Movie Recommender System")
st.caption("Content-based movie recommendations powered by movie metadata.")

try:
    movies, similarity = load_artifacts()
except FileNotFoundError:
    st.error("A required model artifact is missing. Run the model-building script first.")
    st.stop()

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].dropna().sort_values().unique(),
)

if st.button("✨ Recommend", type="primary"):
    results = recommend(selected_movie, movies, similarity)

    if not results:
        st.warning("No recommendations were found.")
        st.stop()

    st.subheader(f"Because you liked **{selected_movie}**")

    columns = st.columns(5)

    for column, result in zip(columns, results):
        with column:
            try:
                poster = fetch_poster(result["movie_id"])
            except requests.RequestException:
                poster = None

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.info("Poster unavailable")

            st.markdown(f"**{result['title']}**")
            st.caption(f"Similarity · {result['score']:.3f}")
