"""Multi-part catalog analysis: trends, mix, ratings, durations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features import explode_countries, explode_genres


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """% missing per column (coursework Task 1 theme)."""
    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    out = pct[pct > 0].rename("missing_pct").reset_index()
    out.columns = ["column", "missing_pct"]
    return out


def type_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Movies vs TV Shows counts and share (Task 2)."""
    counts = df["type"].value_counts()
    out = counts.rename("count").to_frame()
    out["share"] = out["count"] / out["count"].sum()
    return out.reset_index(names="type")


def titles_added_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly adds, overall and by type (Task 3 + bonus)."""
    work = df.dropna(subset=["date_added"]).copy()
    work["month"] = work["date_added"].dt.to_period("M").astype(str)
    overall = work.groupby("month").size().rename("all").to_frame()
    by_type = (
        work.groupby(["month", "type"]).size().unstack(fill_value=0)
    )
    return overall.join(by_type, how="outer").fillna(0).astype(int).reset_index()


def release_year_summary(df: pd.DataFrame, year_min: int = 1980) -> dict[str, Any]:
    """Release-year distribution stats (Task 4)."""
    years = df["release_year"].dropna()
    recent = years[years >= year_min]
    return {
        "describe": years.describe().to_dict(),
        "share_since_year_min": float((years >= year_min).mean()),
        "year_min_focus": year_min,
        "n_recent": int(recent.shape[0]),
    }


def top_countries(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Exploded country counts (Task 5)."""
    countries = explode_countries(df)
    counts = countries["country"].value_counts().head(n)
    return pd.DataFrame({"country": counts.index.astype(str), "count": counts.to_numpy()})


def rating_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Crosstab rating × type (Task 6)."""
    work = df.dropna(subset=["rating"]).copy()
    ct = pd.crosstab(work["rating"], work["type"])
    ct["Total"] = ct.sum(axis=1)
    ct = ct.sort_values("Total", ascending=False)
    return ct.reset_index()


def rating_band_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse Kids / Teens / Mature bands."""
    work = df.dropna(subset=["rating_band"])
    counts = work["rating_band"].value_counts()
    out = counts.rename("count").to_frame()
    out["share"] = out["count"] / out["count"].sum()
    return out.reset_index(names="rating_band")


def top_genres(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Exploded listed_in counts (Task 7)."""
    genres = explode_genres(df)
    counts = genres["genre"].value_counts().head(n)
    return pd.DataFrame({"genre": counts.index.astype(str), "count": counts.to_numpy()})


def genre_trends_by_year_added(
    df: pd.DataFrame, top_n: int = 6, year_min: int = 2015
) -> pd.DataFrame:
    """Share of top genres among titles added each year."""
    genres = explode_genres(df).dropna(subset=["year_added"])
    genres = genres[genres["year_added"] >= year_min]
    top = genres["genre"].value_counts().head(top_n).index.tolist()
    g = genres[genres["genre"].isin(top)]
    yearly = g.groupby(["year_added", "genre"]).size().rename("count").reset_index()
    totals = genres.groupby("year_added").size().rename("year_total")
    yearly = yearly.merge(totals, on="year_added")
    yearly["share"] = yearly["count"] / yearly["year_total"]
    return yearly


def movie_duration_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Movie minutes describe (Task 8)."""
    mins = df.loc[df["type"] == "Movie", "movie_minutes"].dropna()
    return mins.describe().to_dict()


def tv_seasons_summary(df: pd.DataFrame) -> pd.DataFrame:
    """TV season count distribution (Task 9)."""
    seasons = df.loc[df["type"] == "TV Show", "tv_seasons"].dropna()
    counts = seasons.value_counts().sort_index()
    return pd.DataFrame({"seasons": counts.index.astype(float), "count": counts.to_numpy()})


def movie_year_duration_corr(df: pd.DataFrame, year_min: int = 1970) -> dict[str, float]:
    """Correlation release_year vs movie_minutes (Task 10)."""
    movies = df[
        (df["type"] == "Movie")
        & (df["release_year"] >= year_min)
        & df["movie_minutes"].notna()
    ]
    if movies.empty:
        return {"corr": float("nan"), "n": 0}
    corr = movies[["release_year", "movie_minutes"]].corr().iloc[0, 1]
    return {"corr": float(corr), "n": int(len(movies))}


def catalog_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Stakeholder one-pager KPIs."""
    return {
        "n_titles": int(len(df)),
        "n_movies": int((df["type"] == "Movie").sum()),
        "n_tv": int((df["type"] == "TV Show").sum()),
        "year_added_min": None if df["year_added"].dropna().empty else int(df["year_added"].min()),
        "year_added_max": None if df["year_added"].dropna().empty else int(df["year_added"].max()),
        "release_year_min": int(df["release_year"].min()),
        "release_year_max": int(df["release_year"].max()),
        "pct_with_country": float(df["country"].notna().mean()),
        "pct_with_director": float(df["director"].notna().mean()),
        "pct_rated": float(df["rating"].notna().mean()),
        "pct_rating_band": float(df["rating_band"].notna().mean()),
    }


def run_analysis(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Bundle all analysis tables / scalars for reporting."""
    a = cfg.get("analysis", {}) or {}
    top_g = int(a.get("top_genres", 12))
    top_c = int(a.get("top_countries", 10))
    year_min = int(a.get("year_min", 2000))
    return {
        "kpis": catalog_kpis(df),
        "missingness": missingness_table(df).to_dict(orient="records"),
        "type_mix": type_mix(df).to_dict(orient="records"),
        "rating_band_mix": rating_band_mix(df).to_dict(orient="records"),
        "top_countries": top_countries(df, n=top_c).to_dict(orient="records"),
        "top_genres": top_genres(df, n=top_g).to_dict(orient="records"),
        "movie_duration": movie_duration_summary(df),
        "tv_seasons_head": tv_seasons_summary(df).head(10).to_dict(orient="records"),
        "release_year": release_year_summary(df, year_min=1980),
        "movie_year_duration_corr": movie_year_duration_corr(df),
        "titles_added_tail": titles_added_over_time(df).tail(12).to_dict(orient="records"),
        "genre_trends": genre_trends_by_year_added(df, year_min=year_min).to_dict(orient="records"),
    }
