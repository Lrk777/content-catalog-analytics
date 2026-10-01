"""Feature engineering for genres, duration, country, and ratings."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# Stable genre vocabulary for multi-hot features (common Netflix listed_in labels)
CORE_GENRES = [
    "Dramas",
    "Comedies",
    "Documentaries",
    "Action & Adventure",
    "International Movies",
    "International TV Shows",
    "Children & Family Movies",
    "Thrillers",
    "Crime TV Shows",
    "Kids' TV",
    "Romantic Movies",
    "Horror Movies",
    "Stand-Up Comedy",
    "TV Dramas",
    "TV Comedies",
    "Independent Movies",
    "Sci-Fi & Fantasy",
    "Music & Musicals",
    "Reality TV",
    "Anime Series",
]


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (title, genre) for mix analysis."""
    rows = []
    for _, r in df.iterrows():
        listed = r.get("listed_in")
        if pd.isna(listed) or not str(listed).strip():
            continue
        for g in str(listed).split(","):
            g = g.strip()
            if g:
                rows.append(
                    {
                        "show_id": r["show_id"],
                        "type": r["type"],
                        "release_year": r["release_year"],
                        "year_added": r.get("year_added"),
                        "rating_band": r.get("rating_band"),
                        "genre": g,
                    }
                )
    return pd.DataFrame(rows)


def explode_countries(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (title, country) for geographic mix."""
    rows = []
    for _, r in df.iterrows():
        country = r.get("country")
        if pd.isna(country) or not str(country).strip():
            continue
        for c in str(country).split(","):
            c = c.strip()
            if c:
                rows.append(
                    {
                        "show_id": r["show_id"],
                        "type": r["type"],
                        "release_year": r["release_year"],
                        "year_added": r.get("year_added"),
                        "country": c,
                    }
                )
    return pd.DataFrame(rows)


def genre_multihot(df: pd.DataFrame, genres: list[str] | None = None) -> pd.DataFrame:
    """Binary flags for core genres (titles can have multiple)."""
    genres = genres or CORE_GENRES
    out = pd.DataFrame(index=df.index)
    listed = df["listed_in"].fillna("").astype(str)
    for g in genres:
        # Match whole genre tokens after split to avoid substring false positives
        col = f"genre_{g.lower().replace(' ', '_').replace('&', 'and').replace(chr(39), '')}"
        out[col] = listed.apply(
            lambda s, target=g: int(target in [x.strip() for x in s.split(",") if x.strip()])
        )
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modeling matrix for rating-band classification.
    Drops rows without a known rating_band.
    """
    base = df.dropna(subset=["rating_band"]).copy()
    genres = genre_multihot(base)
    feat = pd.DataFrame(
        {
            "is_movie": base["is_movie"].astype(int),
            "release_year": base["release_year"].astype(int),
            "year_added": base["year_added"].fillna(base["release_year"]).astype(int),
            "n_genres": base["n_genres"].fillna(0).astype(int),
            "n_countries": base["n_countries"].fillna(0).astype(int),
            "has_director": base["has_director"].astype(int),
            "has_cast": base["has_cast"].astype(int),
            "duration_minutes": pd.to_numeric(base["duration_minutes"], errors="coerce"),
            "n_seasons": pd.to_numeric(base["n_seasons"], errors="coerce"),
        },
        index=base.index,
    )
    # Movies: fill seasons with 0; TV: fill minutes with median movie length proxy later
    feat["duration_minutes"] = feat["duration_minutes"].fillna(0.0)
    feat["n_seasons"] = feat["n_seasons"].fillna(0.0)
    # Primary country frequency encoding (train-safe: computed outside split in model.py)
    feat["primary_country"] = base["primary_country"].fillna("Unknown").astype(str)
    out = pd.concat([feat, genres], axis=1)
    out["rating_band"] = base["rating_band"].astype(str)
    out["show_id"] = base["show_id"].astype(str)
    return out.reset_index(drop=True)


def frequency_encode(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Map categories to train frequencies; unseen → 0."""
    counts = train.value_counts(normalize=True)
    tr = train.map(counts).fillna(0.0).to_numpy(dtype=float)
    te = test.map(counts).fillna(0.0).to_numpy(dtype=float)
    return tr, te


def top_counts(series: pd.Series, n: int = 10) -> pd.DataFrame:
    c = Counter(series.dropna().astype(str))
    items = c.most_common(n)
    return pd.DataFrame(items, columns=["label", "count"])
