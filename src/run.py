"""One-command pipeline: clean → analyze → model → figures → metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analyze import run_analysis
from src.data import load_clean, load_config, repo_root
from src.model import cluster_content, train_rating_band_model
from src.plot import write_all_figures


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def findings_markdown(metrics: dict[str, Any]) -> str:
    k = metrics["analysis"]["kpis"]
    m = metrics["model"]
    c = metrics["clustering"]
    base = m["baseline"]
    mod = m["model"]
    lines = [
        "# Catalog analytics — findings snapshot",
        "",
        f"- **{k['n_titles']:,}** cleaned titles ({k['n_movies']:,} movies / {k['n_tv']:,} TV shows).",
        f"- Rating-band coverage: **{100 * k['pct_rating_band']:.1f}%** of titles mapped to Kids / Teens / Mature.",
        f"- **Rating-band model:** RandomForest accuracy **{mod['accuracy']:.3f}** "
        f"(macro-F1 **{mod['macro_f1']:.3f}**) vs majority baseline "
        f"**{base['accuracy']:.3f}** (macro-F1 **{base['macro_f1']:.3f}**).",
        f"- **Clustering:** k={c['n_clusters']}, silhouette **{c['silhouette']:.3f}** "
        f"(ARI vs rating band **{c['ari_vs_rating_band']:.3f}** — descriptive only).",
        "",
        "See README for stakeholder narrative and Limitations.",
    ]
    return "\n".join(lines) + "\n"


def run(config_path: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg = load_config(config_path) if config_path else load_config()
    seed = int(cfg.get("seed", 42))

    df = load_clean(cfg, root=root)
    analysis = run_analysis(df, cfg)

    model_result = train_rating_band_model(df, cfg, seed=seed)
    # Strip non-JSON / heavy objects for metrics file
    model_public = {
        k: v
        for k, v in model_result.items()
        if k not in {"clf", "y_test", "y_pred"}
    }
    cluster_result = cluster_content(df, cfg, seed=seed)

    figures_dir = root / cfg["reports"]["figures_dir"]
    latest_dir = root / cfg["reports"]["latest_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    write_all_figures(df, figures_dir, model_result=model_result, cfg=cfg)

    metrics = {
        "seed": seed,
        "data_source": {
            "name": cfg["data"].get("source_name"),
            "url": cfg["data"].get("source_url"),
            "raw_csv": cfg["data"]["raw_csv"],
            "n_rows_clean": int(len(df)),
        },
        "analysis": analysis,
        "model": model_public,
        "clustering": cluster_result,
        "coursework_themes": [
            "missingness overview",
            "Movies vs TV mix",
            "titles added over time",
            "release-year distribution",
            "top countries (exploded)",
            "rating × type",
            "top genres (exploded)",
            "movie duration + TV seasons",
            "release year vs movie minutes",
            "rating-band prediction + content clustering (portfolio extension)",
        ],
    }
    metrics = _json_ready(metrics)

    metrics_path = latest_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    findings_path = latest_dir / "findings.md"
    findings_path.write_text(findings_markdown(metrics))

    # snapshot config used
    with (latest_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    snap_dir = root / "reports" / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Store a slim expected metrics for tests (accuracy / silhouette tolerances)
    slim = {
        "model_accuracy": metrics["model"]["model"]["accuracy"],
        "baseline_accuracy": metrics["model"]["baseline"]["accuracy"],
        "macro_f1": metrics["model"]["model"]["macro_f1"],
        "silhouette": metrics["clustering"]["silhouette"],
        "n_rows_clean": metrics["data_source"]["n_rows_clean"],
    }
    with (snap_dir / "expected_metrics.json").open("w") as f:
        json.dump(slim, f, indent=2)

    print(f"Cleaned titles: {len(df):,}")
    print(
        f"RF accuracy={metrics['model']['model']['accuracy']:.3f} "
        f"macro-F1={metrics['model']['model']['macro_f1']:.3f} | "
        f"baseline acc={metrics['model']['baseline']['accuracy']:.3f}"
    )
    print(
        f"Clusters k={metrics['clustering']['n_clusters']} "
        f"silhouette={metrics['clustering']['silhouette']:.3f}"
    )
    print(f"Wrote {metrics_path}")
    print(f"Figures → {figures_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run content-catalog-analytics pipeline")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
