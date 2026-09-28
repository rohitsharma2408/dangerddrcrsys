# Dataset

This project uses the TMDB 5000 Movie Dataset.

Place these files in this directory before running `scripts/build_model.py`:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The dataset is not committed to this repository.

After adding the files, run:

```bash
python scripts/build_model.py
```

This creates:

- `movie_dict.pkl`
- `similarity_1.pkl`
