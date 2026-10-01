# Content Catalog Analytics

Streaming **catalog EDA → features → rating-band modeling** on a public Netflix titles extract. Expanded from a Pandas/Matplotlib Netflix homework into a reproducible portfolio package (modules, tests, stakeholder findings) — not a notebook dump.

## Problem

What does a large SVOD catalog look like by type, geography, genre, and maturity rating — and can catalog metadata alone predict a coarse **Kids / Teens / Mature** band better than a majority-class baseline?

## Data source

| | |
|--|--|
| Dataset | `netflix_titles.csv` (~7.8k titles) |
| Bundle | Checked in at `data/raw/netflix_titles.csv` |
| Public mirror | [TidyTuesday 2021-04-20](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-04-20) |
| Upstream | Flixable catalog scrape, widely redistributed as Kaggle *Netflix Movies and TV Shows* (Shivam Bansal) |

**Ethical note:** public catalog metadata only. Not affiliated with Netflix. No user viewing histories or private scrapes.

**Coursework lineage:** themes and cleaning steps align with York-style Assignment 1 (*Netflix Titles EDA + Visualization*) — missingness, type mix, adds-over-time, release years, exploded countries/genres, rating×type, movie minutes, TV seasons, year-vs-duration — then extended with rating bands, modeling, clustering, and packaged reports. Source notebook (private store): `internal/coursework-sources/netflix-eda/`.

## Demo

```text
$ python -m src.run --config configs/default.yaml

Cleaned titles: 7,787
RF accuracy=0.692 macro-F1=0.720 | baseline acc=0.465
Clusters k=4 silhouette=0.145
Wrote reports/latest/metrics.json
```

Figures: `reports/figures/`. Snapshot narrative: `reports/latest/findings.md`.

## Method

1. **Cleaning (homework-aligned + hardened):** parse `date_added`; extract `duration_num` → `movie_minutes` / `tv_seasons`; fix duration-leaked ratings; map ratings → **Kids / Teens / Mature**; primary country/genre + counts.
2. **EDA parts:** missingness; Movies vs TV; monthly adds (with type split); release-year hist; top countries/genres (explode); rating×type stacked; duration & seasons; year vs minutes scatter; genre-share trends by year added.
3. **Features:** multi-hot core genres, duration/season signals, country frequency encoding (train-fitted), director/cast flags.
4. **Model:** stratified holdout RandomForest vs **majority-class baseline**; confusion matrix + impurity importances.
5. **Insight add-on:** KMeans (k=4) on scaled features; silhouette + ARI-vs-band as a *weak* external reference only (not supervised labels).

Config: `configs/default.yaml` (seed **42**).

## Results

Checked-in run (seed **42**, stratified 75/25 holdout):

| Metric | Value |
|--------|------:|
| Cleaned titles | **7,787** (69% movies / 31% TV) |
| Majority baseline accuracy | 0.465 (always *Mature*) |
| Majority baseline macro-F1 | 0.212 |
| **RandomForest accuracy** | **0.692** |
| **RandomForest macro-F1** | **0.720** |
| KMeans silhouette (k=4) | 0.145 |
| ARI vs rating band | 0.048 (weak — expected) |

**Catalog facts (same run):** top countries US / India / UK; top genre credits International Movies, Dramas, Comedies; movie minutes vs release year correlation ≈ **−0.24** (newer catalog movies slightly shorter on average). Top RF importances: Kids’ TV, Children & Family Movies, duration minutes.

**Talk track:** Metadata beats a majority baseline by ~23 accuracy points and lifts macro-F1 a lot because Kids is rare but genre-tagged. Clusters are soft (low silhouette) and barely align with maturity bands — useful honesty for interviews.

## Stack

- Python 3.11+
- pandas, NumPy, scikit-learn, matplotlib, seaborn, PyYAML, pytest

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m src.run --config configs/default.yaml
pytest -q
```

## Project layout

```text
configs/default.yaml
data/raw/netflix_titles.csv   # public extract + README attribution
data/processed/               # cleaned CSV written by pipeline
src/data.py                   # load + clean
src/features.py               # genres / country / model frame
src/analyze.py                # multi-part EDA tables
src/model.py                  # rating-band RF + KMeans
src/plot.py                   # figures
src/run.py                    # one-command pipeline
reports/figures/              # stakeholder plots
reports/latest/metrics.json
tests/test_core.py
```

## Limitations

- **Point-in-time catalog** (~2021 TidyTuesday freeze), not a live Netflix API and not viewing demand.
- **Rating bands** collapse heterogeneous regional labels; some titles lack usable ratings.
- **Country field** is production/credit geography, often multi-valued; exploding double-counts co-productions by design.
- **Model** predicts policy maturity from metadata — not IMDb score, popularity, or completion rate.
- **Clustering ARI vs band** is descriptive only; do not read it as cluster “accuracy.”

## License

MIT for original project code. Dataset remains subject to its upstream public redistribution terms; cite TidyTuesday / Flixable–Kaggle lineage if you republish figures.
