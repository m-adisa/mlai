# Project 1.2 — Customer Segmentation

Phase 1 project. Cluster Olist marketplace customers on purchase frequency, spend, and category preference using K-Means, PCA, and DBSCAN.

## Objective

Segment buyers into behaviorally distinct groups usable for targeting/retention decisions, and compare three unsupervised approaches (K-Means, PCA-assisted K-Means, DBSCAN) on the same feature space — matching the algorithms specified for this project in the [curriculum](https://github.com/m-adisa/mlai/blob/main/README.md).

## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100k orders, 2016–2018, multiple Brazilian marketplaces. 9 relational CSVs; tables used:

| Table | Used for |
|---|---|
| `olist_customers_dataset` | customer identity, location |
| `olist_orders_dataset` | order timestamps (recency/frequency) |
| `olist_order_items_dataset` | price, freight (monetary) |
| `olist_order_reviews_dataset` | review scores |
| `olist_products_dataset` + `olist_product_category_name_translation` | category (translated to English) |

**Identity key:** `customer_id` is per-order; `customer_unique_id` is the true customer across orders. All frequency/aggregation logic keys on `customer_unique_id`.

**Repeat-purchase reality:** ~97% of unique customers have exactly one order — frequency has near-zero variance across the full base. Handled as two parallel tracks rather than picked around:

1. **Full base** — all customers; frequency mostly collapses to 1, so segmentation is driven mainly by spend/category/experience features.
2. **Repeat customers only** (≥2 orders, ~3% of base) — frequency is a live signal here.

Both tracks run through the same pipeline; results compared in the notebook (cluster count, stability, which features drive separation in each).

## Features

Computed per `customer_unique_id`:

**RFM**
- Recency — days since last order (relative to dataset max date)
- Frequency — order count (informative mainly on the repeat-only track)
- Monetary — total spend, `log1p`-transformed

**Category preference**
- Spend-share vector across top-10 `product_category_name_english` categories + an `other` bucket (11 dims). Top-10 threshold to be confirmed against actual category volume concentration once EDA runs.

**Marketplace extras**
- Avg review score
- Avg delivery delay = actual delivery date − estimated delivery date (negative = early)

## Preprocessing

1. `log1p` on monetary
2. `StandardScaler` on the full feature set
3. `PCA` for the PCA-assisted track and for 2D visualization of all cluster results

## Algorithms

- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — k chosen via elbow (inertia) + [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), both reported
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — dimensionality reduction before K-Means and for visualization
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) — `eps` from a k-distance plot, `min_samples = 2 × n_features`, per the heuristic in the original DBSCAN paper ([Ester et al., 1996](https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695-00/papers/00153.pdf), refined in [Sander et al., 1998](https://static.aminer.org/pdf/PDF/000/307/216/a_density_based_approach_for_clustering_spatial_database.pdf)). Run on **both** the full scaled feature space and the PCA-reduced space — density-based distance metrics degrade at ~16 dimensions, so the full-space run alone would stack the deck against DBSCAN. Both results reported; `eps`/`min_samples` re-derived separately per space (k-distance plot and dimensionality both change)

## Evaluation

- [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [Davies-Bouldin](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html), [Calinski-Harabasz](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) — computed across all three algorithms/tracks as the common cross-algorithm yardstick. Caveat carried into the comparison, not hidden: all three are centroid/distance-based and implicitly reward convex clusters, so they structurally favor K-Means's geometry over DBSCAN's — directionally comparable, not strictly equivalent, across algorithms
- **[DBCV](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)** ([FelSiq/DBCV](https://github.com/FelSiq/DBCV) implementation) — required primary validation metric for DBSCAN specifically, not optional. Scores density-connectedness rather than distance-to-centroid, so it's the correct tool for what DBSCAN actually produces, unlike the three metrics above. Handles noise (`-1`) natively as part of the score — no exclusion needed. Where DBCV and the silhouette/DB/CH verdicts on DBSCAN disagree is reported explicitly, not resolved by picking one
- Qualitative segment profiling — per-cluster RFM/category/review/delivery summary table, PCA 2D scatter
- **Stability check** — bootstrap resample the customer set (e.g. n=20 resamples), re-cluster each, measure label agreement against the full-data clustering via [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html). Reported per algorithm/track.

**DBCV compute constraint.** DBCV builds a mutual-reachability MST over the full pairwise distance graph — effectively O(n²) memory/compute. Feasible directly on the repeat-only track (~3k customers). Not feasible on the full-base track (~96k unique customers) at full size — computed there on a random subsample (~5k–10k, same sampling approach as the stability check), stated explicitly in the results rather than silently subsampled.

**Silhouette/DB/CH noise-point handling** (unaffected by the DBCV addition — these three still need it, DBCV doesn't):
- Noise points excluded before computing all three (standard convention — scoring them would penalize DBSCAN for correctly refusing to force-assign outliers)
- Noise fraction (`% labeled -1`) reported as its own stat alongside the metrics, not folded into them
- If a run leaves <2 clusters after removing noise, or noise fraction exceeds ~50%, metrics are flagged unreliable rather than reported at face value against K-Means

## Project structure

```
Phase_1/Project/
├── README.md
├── data/
│   └── raw/              # gitignored — Olist CSVs, downloaded not committed
├── src/
│   ├── data_loading.py   # download/load + customer_unique_id joins
│   ├── features.py       # RFM, category vector, marketplace extras
│   ├── clustering.py     # K-Means, PCA, DBSCAN, param selection
│   └── evaluation.py     # metrics + stability/ARI harness
└── notebook.ipynb         # EDA, algorithm comparison, full vs repeat-only comparison
```

## Environment

Add to `Pipfile`:
- `kagglehub` — dataset download
- `scipy` — DBSCAN/stability support
- `dbcv` — DBCV metric ([FelSiq/DBCV](https://github.com/FelSiq/DBCV); install via `pip install "git+https://github.com/FelSiq/DBCV"`, or as a Pipfile git source)

`data/raw/` is gitignored; notebook/script downloads on first run via `kagglehub` (instructions in `src/data_loading.py`).

## Resources

- [Olist dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [scikit-learn: silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [scikit-learn: davies_bouldin_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
- [scikit-learn: calinski_harabasz_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html)
- [scikit-learn: adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
- Ester, Kriegel, Sander, Xu (1996), [original DBSCAN paper](https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695-00/papers/00153.pdf)
- Sander, Ester, Kriegel, Xu (1998), [GDBSCAN — min_samples heuristic](https://static.aminer.org/pdf/PDF/000/307/216/a_density_based_approach_for_clustering_spatial_database.pdf)
- Moulavi, Jaskowiak, Campello, Zimek, Sander (2014), [DBCV — density-based clustering validation](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)
- [FelSiq/DBCV — Python DBCV implementation](https://github.com/FelSiq/DBCV)
- [mlai curriculum](https://github.com/m-adisa/mlai/blob/main/README.md)
