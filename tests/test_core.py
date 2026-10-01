"""Tests for cleaning, features, and modeling hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import RATING_BANDS, clean_titles, parse_duration, repo_root
from src.features import build_model_frame, explode_genres, frequency_encode, genre_multihot
from src.model import majority_baseline_predict, train_rating_band_model


def _mini_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3", "s4", "s5"],
            "type": ["Movie", "TV Show", "Movie", "TV Show", "Movie"],
            "title": ["A", "B", "C", "D", "E"],
            "director": ["X", pd.NA, "Y", pd.NA, "Z"],
            "cast": ["a, b", "c", pd.NA, "d", "e"],
            "country": ["United States", "India, United States", "Japan", pd.NA, "United Kingdom"],
            "date_added": ["January 1, 2020", "March 5, 2019", "bad-date", "July 4, 2021", "August 8, 2018"],
            "release_year": [2019, 2018, 2015, 2020, 2010],
            "rating": ["TV-MA", "TV-14", "PG", "TV-Y", "R"],
            "duration": ["98 min", "2 Seasons", "120 min", "1 Season", "90 min"],
            "listed_in": [
                "Dramas, International Movies",
                "International TV Shows, TV Dramas",
                "Children & Family Movies, Comedies",
                "Kids' TV",
                "Action & Adventure, Dramas",
            ],
            "description": ["d1", "d2", "d3", "d4", "d5"],
        }
    )


def test_parse_duration():
    assert parse_duration("98 min") == (98.0, "min")
    assert parse_duration("2 Seasons") == (2.0, "season")
    assert parse_duration(None) == (None, None)


def test_clean_titles_coursework_columns():
    clean = clean_titles(_mini_raw())
    assert "date_added" in clean.columns
    assert pd.api.types.is_datetime64_any_dtype(clean["date_added"])
    assert "movie_minutes" in clean.columns
    assert "tv_seasons" in clean.columns
    assert "duration_num" in clean.columns
    assert clean.loc[clean["type"] == "Movie", "movie_minutes"].notna().all()
    assert clean.loc[clean["type"] == "TV Show", "tv_seasons"].notna().all()
    assert set(clean["rating_band"].dropna()) <= {"Kids", "Teens", "Mature"}
    assert clean["primary_country"].iloc[1] == "India"


def test_rating_band_map_covers_common():
    for r in ("TV-MA", "PG-13", "TV-Y", "R", "G"):
        assert r in RATING_BANDS


def test_genre_multihot_and_explode():
    clean = clean_titles(_mini_raw())
    hot = genre_multihot(clean)
    assert hot.shape[0] == len(clean)
    assert hot.filter(like="genre_").sum().sum() >= 1
    exploded = explode_genres(clean)
    assert len(exploded) >= len(clean)


def test_frequency_encode_unseen_zero():
    tr = pd.Series(["US", "US", "IN", "UK"])
    te = pd.Series(["US", "FR"])
    a, b = frequency_encode(tr, te)
    assert a[0] == pytest.approx(0.5)
    assert b[1] == pytest.approx(0.0)


def test_majority_baseline():
    y = pd.Series(["Mature", "Mature", "Teens", "Mature"])
    pred = majority_baseline_predict(y, 3)
    assert list(pred) == ["Mature", "Mature", "Mature"]


def test_model_frame_drops_unbanded():
    raw = _mini_raw()
    raw.loc[0, "rating"] = "NOT-A-RATING"
    clean = clean_titles(raw)
    assert clean.loc[0, "rating_band"] is pd.NA or pd.isna(clean.loc[0, "rating_band"])
    frame = build_model_frame(clean)
    assert frame["rating_band"].isna().sum() == 0
    assert len(frame) == clean["rating_band"].notna().sum()
    assert "NOT-A-RATING" not in set(frame.merge(
        clean[["show_id", "rating"]], on="show_id", how="left"
    )["rating"])


def test_train_on_synthetic_is_deterministic():
    rng = np.random.default_rng(0)
    n = 240
    bands = rng.choice(["Kids", "Teens", "Mature"], size=n, p=[0.2, 0.35, 0.45])
    rows = []
    for i, band in enumerate(bands):
        rating = {"Kids": "TV-Y", "Teens": "TV-14", "Mature": "TV-MA"}[band]
        typ = "Movie" if i % 2 == 0 else "TV Show"
        dur = f"{90 + (i % 40)} min" if typ == "Movie" else f"{1 + (i % 3)} Seasons"
        genre = {
            "Kids": "Kids' TV, Children & Family Movies",
            "Teens": "TV Dramas, Comedies",
            "Mature": "Dramas, Thrillers",
        }[band]
        rows.append(
            {
                "show_id": f"s{i}",
                "type": typ,
                "title": f"T{i}",
                "director": "D" if i % 3 else pd.NA,
                "cast": "c",
                "country": rng.choice(["United States", "India", "Japan"]),
                "date_added": "January 1, 2020",
                "release_year": 2000 + (i % 20),
                "rating": rating,
                "duration": dur,
                "listed_in": genre,
                "description": "x",
            }
        )
    clean = clean_titles(pd.DataFrame(rows))
    cfg = {"model": {"test_size": 0.25, "rf": {"n_estimators": 40, "max_depth": 6, "min_samples_leaf": 2}}}
    a = train_rating_band_model(clean, cfg, seed=42)
    b = train_rating_band_model(clean, cfg, seed=42)
    assert a["model"]["accuracy"] == b["model"]["accuracy"]
    assert a["model"]["accuracy"] >= a["baseline"]["accuracy"] - 1e-9


def test_snapshot_metrics_if_present():
    """When full pipeline has been run, snapshot stays in a sensible range."""
    snap = repo_root() / "reports" / "snapshots" / "expected_metrics.json"
    if not snap.exists():
        pytest.skip("pipeline snapshot not generated yet")
    data = json.loads(snap.read_text())
    assert data["n_rows_clean"] > 5000
    assert 0.4 <= data["model_accuracy"] <= 1.0
    assert data["model_accuracy"] + 1e-9 >= data["baseline_accuracy"]
    assert -1.0 <= data["silhouette"] <= 1.0
