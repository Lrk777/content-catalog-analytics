"""Stakeholder-ready figures for catalog analytics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analyze import (
    missingness_table,
    rating_by_type,
    titles_added_over_time,
    top_countries,
    top_genres,
    tv_seasons_summary,
)
from src.features import explode_genres

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, path: Path) -> None:
    miss = missingness_table(df)
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ord_ = miss.sort_values("missing_pct")
    ax.barh(ord_["column"], ord_["missing_pct"], color="#e07a5f", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    _save(fig, path)


def plot_type_mix(df: pd.DataFrame, path: Path) -> None:
    counts = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#3d5a80", "#ee6c4d"]
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors[: len(counts)])
    ax.set_title("Catalog mix: Movies vs TV Shows")
    _save(fig, path)


def plot_titles_added(df: pd.DataFrame, path: Path) -> None:
    ts = titles_added_over_time(df)
    if ts.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(ts["month"])
    ax.plot(x, ts["all"], color="#293241", lw=2, label="All")
    if "Movie" in ts.columns:
        ax.plot(x, ts["Movie"], color="#3d5a80", lw=1.5, alpha=0.85, label="Movie")
    if "TV Show" in ts.columns:
        ax.plot(x, ts["TV Show"], color="#ee6c4d", lw=1.5, alpha=0.85, label="TV Show")
    ax.set_title("Titles added to catalog over time")
    ax.set_xlabel("Month added")
    ax.set_ylabel("Titles")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_release_years(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    years = df["release_year"].dropna()
    ax.hist(years, bins=40, color="#98c1d9", edgecolor="black", linewidth=0.3)
    ax.set_xlim(1980, years.max() + 1)
    ax.set_title("Release-year distribution (focus ≥ 1980)")
    ax.set_xlabel("Release year")
    ax.set_ylabel("Titles")
    _save(fig, path)


def plot_top_countries(df: pd.DataFrame, path: Path, n: int = 10) -> None:
    top = top_countries(df, n=n)
    # column may be named country or from value_counts reset
    label_col = "country" if "country" in top.columns else top.columns[0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ord_ = top.sort_values("count")
    ax.barh(ord_[label_col], ord_["count"], color="#2a9d8f", edgecolor="black", linewidth=0.4)
    ax.set_title(f"Top {n} producing countries (exploded)")
    ax.set_xlabel("Titles (credit shares; multi-country titles count in each)")
    _save(fig, path)


def plot_rating_by_type(df: pd.DataFrame, path: Path) -> None:
    ct = rating_by_type(df)
    if ct.empty:
        return
    plot_df = ct.set_index("rating").drop(columns=["Total"], errors="ignore")
    # Keep top 12 ratings by volume
    plot_df = plot_df.head(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax, color=["#3d5a80", "#ee6c4d"])
    ax.set_title("Content rating by type")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Titles")
    ax.legend(title="Type", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    _save(fig, path)


def plot_top_genres(df: pd.DataFrame, path: Path, n: int = 12) -> None:
    top = top_genres(df, n=n)
    label_col = "genre" if "genre" in top.columns else top.columns[0]
    fig, ax = plt.subplots(figsize=(9, 6))
    ord_ = top.sort_values("count")
    ax.barh(ord_[label_col], ord_["count"], color="#e76f51", edgecolor="black", linewidth=0.4)
    ax.set_title(f"Top {n} genres / categories (exploded listed_in)")
    ax.set_xlabel("Title–genre credits")
    _save(fig, path)


def plot_movie_duration(df: pd.DataFrame, path: Path) -> None:
    mins = df.loc[df["type"] == "Movie", "movie_minutes"].dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.hist(mins, bins=40, color="#264653", edgecolor="black", linewidth=0.3)
    ax1.set_title("Movie duration distribution")
    ax1.set_ylabel("Frequency")
    ax2.boxplot(mins, vert=False, patch_artist=True, boxprops=dict(facecolor="#2a9d8f"))
    ax2.set_xlabel("Minutes")
    ax2.set_yticks([])
    _save(fig, path)


def plot_tv_seasons(df: pd.DataFrame, path: Path) -> None:
    seasons = tv_seasons_summary(df).head(10)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(seasons["seasons"].astype(int).astype(str), seasons["count"], color="#f4a261", edgecolor="black", linewidth=0.4)
    ax.set_title("TV shows by number of seasons (top 10 season counts)")
    ax.set_xlabel("Seasons")
    ax.set_ylabel("Shows")
    _save(fig, path)


def plot_year_vs_duration(df: pd.DataFrame, path: Path) -> None:
    movies = df[(df["type"] == "Movie") & (df["release_year"] >= 1970)].dropna(
        subset=["movie_minutes", "release_year"]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(
        movies["release_year"],
        movies["movie_minutes"],
        alpha=0.25,
        s=18,
        c="#c1121f",
        edgecolors="none",
    )
    ax.set_title("Movie duration vs release year")
    ax.set_xlabel("Release year")
    ax.set_ylabel("Minutes")
    _save(fig, path)


def plot_genre_trends(df: pd.DataFrame, path: Path, year_min: int = 2015) -> None:
    genres = explode_genres(df).dropna(subset=["year_added"])
    genres = genres[genres["year_added"] >= year_min]
    top = genres["genre"].value_counts().head(5).index.tolist()
    g = genres[genres["genre"].isin(top)]
    yearly = g.groupby(["year_added", "genre"]).size().rename("count").reset_index()
    totals = genres.groupby("year_added").size().rename("year_total")
    yearly = yearly.merge(totals, on="year_added")
    yearly["share"] = yearly["count"] / yearly["year_total"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, part in yearly.groupby("genre"):
        ax.plot(part["year_added"], part["share"], marker="o", lw=2, label=name)
    ax.set_title(f"Genre share among titles added ({year_min}+)")
    ax.set_xlabel("Year added")
    ax.set_ylabel("Share of title–genre credits")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def plot_confusion(labels: list[str], cm: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    arr = np.asarray(cm)
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted rating band")
    ax.set_ylabel("True rating band")
    ax.set_title("Rating-band classifier — confusion matrix")
    _save(fig, path)


def plot_importances(importances: dict[str, float], path: Path) -> None:
    s = pd.Series(importances).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(s.index, s.values, color="#457b9d", edgecolor="black", linewidth=0.3)
    ax.set_title("Top features — rating-band RandomForest")
    ax.set_xlabel("Impurity importance")
    _save(fig, path)


def write_all_figures(
    df: pd.DataFrame,
    figures_dir: Path,
    model_result: dict | None = None,
    cfg: dict | None = None,
) -> list[str]:
    cfg = cfg or {}
    a = cfg.get("analysis", {}) or {}
    paths = []
    jobs = [
        ("missingness.png", lambda p: plot_missingness(df, p)),
        ("type_mix.png", lambda p: plot_type_mix(df, p)),
        ("titles_added_over_time.png", lambda p: plot_titles_added(df, p)),
        ("release_year_hist.png", lambda p: plot_release_years(df, p)),
        ("top_countries.png", lambda p: plot_top_countries(df, p, n=int(a.get("top_countries", 10)))),
        ("rating_by_type.png", lambda p: plot_rating_by_type(df, p)),
        ("top_genres.png", lambda p: plot_top_genres(df, p, n=int(a.get("top_genres", 12)))),
        ("movie_duration.png", lambda p: plot_movie_duration(df, p)),
        ("tv_seasons.png", lambda p: plot_tv_seasons(df, p)),
        ("year_vs_duration.png", lambda p: plot_year_vs_duration(df, p)),
        ("genre_trends.png", lambda p: plot_genre_trends(df, p, year_min=int(a.get("year_min", 2000)))),
    ]
    for name, fn in jobs:
        out = figures_dir / name
        fn(out)
        paths.append(str(out))
    if model_result:
        plot_confusion(model_result["labels"], model_result["confusion_matrix"], figures_dir / "confusion_matrix.png")
        paths.append(str(figures_dir / "confusion_matrix.png"))
        plot_importances(model_result["top_importances"], figures_dir / "feature_importances.png")
        paths.append(str(figures_dir / "feature_importances.png"))
    return paths
