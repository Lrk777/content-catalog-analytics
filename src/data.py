"""Load and clean the public Netflix titles catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Content ratings → coarse audience bands used as modeling target
RATING_BANDS: dict[str, str] = {
    "G": "Kids",
    "TV-Y": "Kids",
    "TV-G": "Kids",
    "TV-Y7": "Kids",
    "TV-Y7-FV": "Kids",
    "PG": "Teens",
    "TV-PG": "Teens",
    "PG-13": "Teens",
    "TV-14": "Teens",
    "R": "Mature",
    "TV-MA": "Mature",
    "NC-17": "Mature",
    "NR": "Mature",
    "UR": "Mature",
    "TV-NR": "Mature",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (repo_root() / "configs" / "default.yaml")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the Flixable/Kaggle-style titles CSV."""
    df = pd.read_csv(csv_path)
    expected = {
        "show_id",
        "type",
        "title",
        "director",
        "cast",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw catalog missing columns: {sorted(missing)}")
    return df


def parse_duration(value: object) -> tuple[float | None, str | None]:
    """Parse '90 min' / '2 Seasons' → (numeric, unit)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    parts = text.split()
    if len(parts) < 2:
        return None, None
    try:
        num = float(parts[0])
    except ValueError:
        return None, None
    unit = parts[1].lower().rstrip("s")  # min / season
    if unit.startswith("min"):
        return num, "min"
    if unit.startswith("season"):
        return num, "season"
    return num, unit


def clean_titles(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust cleaning for catalog analytics:
    - trim strings, normalize type/rating
    - parse date_added and duration
    - explode primary country / primary genre helpers (kept as columns)
    - drop unusable rows (no type / no title / no release_year)
    """
    df = raw.copy()
    for col in ("show_id", "type", "title", "director", "cast", "country", "rating", "duration", "listed_in", "description"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "NA"]), col] = pd.NA

    df["type"] = df["type"].str.title()
    df = df[df["type"].isin(["Movie", "Tv Show"])].copy()
    df["type"] = df["type"].replace({"Tv Show": "TV Show"})

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["title", "release_year"]).copy()
    df["release_year"] = df["release_year"].astype(int)

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    parsed = df["duration"].map(parse_duration)
    df["duration_value"] = [p[0] for p in parsed]
    df["duration_unit"] = [p[1] for p in parsed]
    # Coursework-aligned names (Assignment 1): movie_minutes / tv_seasons
    df["duration_num"] = df["duration_value"]
    df["movie_minutes"] = np.where(
        df["type"] == "Movie", df["duration_value"], np.nan
    )
    df["tv_seasons"] = np.where(
        df["type"] == "TV Show", df["duration_value"], np.nan
    )
    df["duration_minutes"] = df["movie_minutes"]
    df["n_seasons"] = df["tv_seasons"]

    # Rating hygiene: some rows have duration leaked into rating (known data quirk)
    bad_rating = df["rating"].fillna("").str.contains(r"min|Season", case=False, regex=True)
    df.loc[bad_rating, "rating"] = pd.NA
    df["rating"] = df["rating"].str.upper()
    df["rating_band"] = df["rating"].map(RATING_BANDS)

    df["primary_country"] = (
        df["country"]
        .fillna("")
        .str.split(",")
        .str[0]
        .str.strip()
        .replace({"": pd.NA})
    )
    df["n_countries"] = df["country"].fillna("").apply(
        lambda s: 0 if not str(s).strip() or str(s) == "<NA>" else len([c for c in str(s).split(",") if c.strip()])
    )
    df["primary_genre"] = (
        df["listed_in"]
        .fillna("")
        .str.split(",")
        .str[0]
        .str.strip()
        .replace({"": pd.NA})
    )
    df["n_genres"] = df["listed_in"].fillna("").apply(
        lambda s: 0 if not str(s).strip() or str(s) == "<NA>" else len([g for g in str(s).split(",") if g.strip()])
    )
    df["has_director"] = df["director"].notna().astype(int)
    df["has_cast"] = df["cast"].notna().astype(int)
    df["is_movie"] = (df["type"] == "Movie").astype(int)

    return df.reset_index(drop=True)


def load_clean(cfg: dict[str, Any] | None = None, root: Path | None = None) -> pd.DataFrame:
    """Load raw CSV, clean, optionally write processed artifact."""
    root = root or repo_root()
    cfg = cfg or load_config()
    raw_path = root / cfg["data"]["raw_csv"]
    processed_path = root / cfg["data"]["processed_csv"]
    raw = load_raw(raw_path)
    clean = clean_titles(raw)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    return clean
