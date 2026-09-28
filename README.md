# Movie Recommendation System

A content-based movie recommendation system built with Python and Streamlit using the TMDB 5000 Movie Dataset.

## Overview

This project recommends movies based on similarity between their metadata.

The pipeline combines:

- Movie overview
- Genres
- Keywords
- Top cast members
- Director

These fields are combined into a single text representation, transformed with CountVectorizer, and compared using cosine similarity.

The Streamlit application lets a user select a movie and returns five similar movies with posters retrieved from TMDB.

## Architecture

```text
TMDB 5000 Dataset
        |
        v
Data Cleaning & Feature Extraction
        |
        v
Metadata Tags
(overview + genres + keywords + cast + director)
        |
        v
Text Preprocessing & Stemming
        |
        v
CountVectorizer
(5,000 features)
        |
        v
Cosine Similarity
        |
        v
Recommendation Engine
        |
        v
Streamlit Application
        |
        v
TMDB Poster API
```

## Methodology

### 1. Data Preparation

The TMDB movies and credits datasets are merged using the movie identifier.

Structured JSON-like fields are parsed to extract useful metadata.

### 2. Feature Engineering

The recommendation profile for each movie is built from:

```text
tags = overview + genres + keywords + cast + director
```

The resulting tags are normalized and stemmed using NLTK's Porter Stemmer.

### 3. Vectorization

CountVectorizer converts the text representation into a numerical feature matrix.

The original model uses:

- Maximum features: 5,000
- English stop-word removal

### 4. Similarity

Cosine similarity is calculated between every movie vector.

For a selected movie, the system ranks all other movies by similarity score and returns the top five.

## Dataset

The project uses the TMDB 5000 Movie Dataset containing movie metadata and credits.

The original notebook was developed in Kaggle and therefore contains Kaggle-specific input paths. The notebook should be run with the dataset available at those paths or updated to point to a local data directory.

The merged dataset contains approximately 4,809 movies, with 4,806 records used in the final recommendation dataframe.

## Project Structure

```text
dangerddrcrsys/
├── app/
│   └── app.py
├── movie_dict.pkl
├── notebooks/
│   └── movie_recommendation.ipynb
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
└── .gitignore
```

The repository currently contains the movie metadata artifact `movie_dict.pkl`. The similarity matrix `similarity_1.pkl` is required by the application and must be generated from the notebook before deployment.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## TMDB API Configuration

The application does not store the TMDB API key in source code.

For local development, set an environment variable:

```bash
# Windows PowerShell
$env:TMDB_API_KEY="your_api_key"

# Linux/macOS
export TMDB_API_KEY="your_api_key"
```

For Streamlit deployment, use Streamlit secrets:

```toml
TMDB_API_KEY = "your_api_key"
```

Never commit `.streamlit/secrets.toml`.

## Running the Application

After generating the required `similarity_1.pkl` artifact:

```bash
streamlit run app/app.py
```

## Model Artifacts

| Artifact | Purpose |
|---|---|
| `movie_dict.pkl` | Serialized movie metadata used by the application |
| `similarity_1.pkl` | Pairwise cosine-similarity matrix required for recommendations |

The similarity matrix is intentionally generated from the notebook rather than fabricated or replaced with a different algorithm.

## Example

For a movie such as `Batman Begins`, the original model produced recommendations including other Batman-related titles because their metadata representations are highly similar.

## Limitations

- Recommendations depend entirely on available metadata.
- The system is content-based and does not learn individual user preferences.
- The similarity matrix scales quadratically with the number of movies.
- Poster display depends on TMDB API availability and a valid API key.
- Duplicate or very similar metadata can result in repetitive recommendations.

## Future Improvements

- Add a reproducible model-generation script.
- Automate creation of `similarity_1.pkl`.
- Add automated tests.
- Improve duplicate-title handling.
- Add richer movie metadata to the interface.
- Add deployment configuration.
- Evaluate alternative text representations such as TF-IDF.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, NLTK, Streamlit, Requests, TMDB API.

## Author

Rohit Sharma
