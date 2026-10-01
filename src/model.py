"""Rating-band classification + content clustering with honest metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import build_model_frame, frequency_encode


BAND_ORDER = ["Kids", "Teens", "Mature"]


def _feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = frame["rating_band"].astype(str)
    drop = {"rating_band", "show_id", "primary_country"}
    # primary_country becomes freq column after encoding
    X = frame.drop(columns=[c for c in drop if c in frame.columns]).copy()
    # keep country freq separate — filled in split helpers
    return X, y, list(X.columns)


def majority_baseline_predict(y_train: pd.Series, n_test: int) -> np.ndarray:
    mode = y_train.mode().iloc[0]
    return np.array([mode] * n_test, dtype=object)


def train_rating_band_model(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """
    Predict Kids / Teens / Mature from catalog features.
    Uses stratified holdout; reports majority baseline + RandomForest.
    """
    mcfg = cfg.get("model", {}) or {}
    frame = build_model_frame(df)
    if frame["rating_band"].nunique() < 2:
        raise ValueError("Need at least two rating bands for classification")

    countries = frame["primary_country"]
    X_num = frame.drop(columns=["rating_band", "show_id", "primary_country"])
    y = frame["rating_band"]

    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X_num,
        y,
        countries,
        test_size=float(mcfg.get("test_size", 0.25)),
        random_state=seed,
        stratify=y,
    )
    c_tr, c_te = frequency_encode(c_train, c_test)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["country_freq"] = c_tr
    X_test["country_freq"] = c_te

    # Majority baseline
    y_base = majority_baseline_predict(y_train, len(y_test))
    base_acc = float(accuracy_score(y_test, y_base))
    base_macro_f1 = float(f1_score(y_test, y_base, average="macro", labels=BAND_ORDER, zero_division=0))

    rf_cfg = mcfg.get("rf", {}) or {}
    clf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 200)),
        max_depth=int(rf_cfg.get("max_depth", 12)),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = [b for b in BAND_ORDER if b in set(y_test) | set(y_pred)]
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    importances = (
        pd.Series(clf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts_train": y_train.value_counts().to_dict(),
        "class_counts_test": y_test.value_counts().to_dict(),
        "baseline": {
            "name": "majority_class",
            "accuracy": base_acc,
            "macro_f1": base_macro_f1,
            "predicted_class": str(y_train.mode().iloc[0]),
        },
        "model": {
            "name": "random_forest",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0)),
            "classification_report": report,
        },
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_importances": importances.round(4).to_dict(),
        "feature_names": list(X_train.columns),
        "clf": clf,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
    }


def cluster_content(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """
    KMeans on scaled genre + duration + year features.
    Silhouette for quality; ARI vs rating_band as weak external check
    (not a supervised claim — bands are not cluster ground truth).
    """
    mcfg = cfg.get("model", {}) or {}
    n_clusters = int(mcfg.get("n_clusters", 4))
    frame = build_model_frame(df)
    # Drop target/id/country string; use numeric + genre flags only
    X = frame.drop(columns=["rating_band", "show_id", "primary_country"])
    # Subsample if huge for silhouette speed (still reproducible)
    if len(X) > 4000:
        X_fit = X.sample(n=4000, random_state=seed)
        bands = frame.loc[X_fit.index, "rating_band"]
    else:
        X_fit = X
        bands = frame["rating_band"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    sil = float(silhouette_score(Xs, labels)) if n_clusters > 1 and len(X_fit) > n_clusters else float("nan")
    ari = float(adjusted_rand_score(bands.fillna("Unknown"), labels))

    # Cluster profiles: mean of key cols
    profile = X_fit.copy()
    profile["cluster"] = labels
    means = profile.groupby("cluster").mean(numeric_only=True)
    sizes = profile["cluster"].value_counts().sort_index()

    # Dominant rating band per cluster (descriptive only)
    band_tab = (
        pd.crosstab(
            pd.Series(labels, name="cluster"),
            bands.reset_index(drop=True),
            normalize="index",
        )
        .round(3)
        .to_dict()
    )

    return {
        "n_clusters": n_clusters,
        "n_fit": int(len(X_fit)),
        "silhouette": sil,
        "ari_vs_rating_band": ari,
        "cluster_sizes": {str(int(k)): int(v) for k, v in sizes.items()},
        "cluster_means_head": means.round(3).to_dict(orient="index"),
        "rating_band_share_by_cluster": {
            str(k): {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in _invert_band_tab(band_tab).items()
        },
        "note": (
            "ARI vs rating_band is a weak external reference only — "
            "clusters are unsupervised genre/duration structure, not band predictors."
        ),
    }


def _invert_band_tab(band_tab: dict) -> dict[str, dict[str, float]]:
    """Convert pandas to_dict() column-oriented crosstab to cluster → band shares."""
    # band_tab from DataFrame.to_dict() is {band: {cluster: share}}
    out: dict[str, dict[str, float]] = {}
    for band, by_cluster in band_tab.items():
        if not isinstance(by_cluster, dict):
            continue
        for cluster, share in by_cluster.items():
            out.setdefault(str(cluster), {})[str(band)] = float(share)
    return out
