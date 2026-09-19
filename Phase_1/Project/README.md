# Project 1.2 — Customer Segmentation

Phase 1 project. Cluster Olist marketplace customers on purchase frequency, spend, and category preference using K-Means, PCA, and DBSCAN.

## Objective

Segment buyers into behaviorally distinct groups usable for targeting/retention decisions, and compare three unsupervised approaches (K-Means on MFA-weighted features, K-Means on PCA-reduced features, DBSCAN) on the same underlying feature space — matching the algorithms specified for this project in the [curriculum](https://github.com/m-adisa/mlai/blob/main/README.md).

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
- Spend-share vector across top-10 `product_category_name_english` categories by spend + an `other` bucket (11 dims). CLR-transformed before scaling/PCA (see Preprocessing) — raw shares are compositional and would distort Euclidean distance otherwise

**Marketplace extras**
- Avg review score
- Avg delivery delay = actual delivery date − estimated delivery date (negative = early)

**Data filtering rules**
- `order_status == 'delivered'` only — undelivered/canceled orders excluded entirely
- Rows with null actual delivery date dropped from the delivery-delay calc (not imputed)
- Customers with zero reviews: review score left out of their profile, not imputed to a fake average

## Preprocessing

1. `log1p` on monetary
2. CLR (centered log-ratio) transform on the category-share vector — raw shares sum to 1, which creates collinearity and breaks the Euclidean geometry PCA/K-Means/DBSCAN assume; CLR removes that (residual rank deficiency of 1 from the closure constraint is left to PCA/SVD to absorb, no separate dimension drop needed)
3. `StandardScaler` per block (RFM; CLR-transformed category; marketplace extras)
4. **Block weighting via Multiple Factor Analysis (MFA)** — the feature space is unbalanced by construction (3 RFM dims vs 11 category dims vs 2 extras); StandardScaler equalizes per-feature variance but not each block's share of total distance, so the 11-dim category block would otherwise dominate Euclidean distance by sheer dimension count. MFA (Escofier & Pagès, 1994) weights each block by the inverse of its own first eigenvalue (from a within-block PCA), so each block contributes equal maximum inertia before the blocks are concatenated — the principled version of block balancing, not a heuristic scalar. Implemented via [`prince.MFA`](https://github.com/MaxHalford/prince) with three groups: RFM, category (CLR), extras
5. Global `PCA` on the MFA-weighted concatenated space — components kept to reach ≥90% explained variance, capped at 8

## Algorithms

- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — run in two spaces for comparison: on the MFA-weighted feature space, and on the MFA+PCA-reduced space. Not a third algorithm — same algorithm, two input spaces, framed that way explicitly to avoid implying otherwise. `k` chosen via elbow (inertia) + [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) independently in each space, both reported
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — dimensionality reduction applied after MFA weighting; components kept to reach ≥90% explained variance, capped at 8; also used for 2D visualization of all cluster results
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) — `eps` from a k-distance plot, `min_samples = 2 × n_features`, per the heuristic in the original DBSCAN paper ([Ester et al., 1996](https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695-00/papers/00153.pdf), refined in [Sander et al., 1998](https://static.aminer.org/pdf/PDF/000/307/216/a_density_based_approach_for_clustering_spatial_database.pdf)). Run on **both** the MFA-weighted full space and the MFA+PCA-reduced space — density-based distance metrics degrade at ~16 dimensions, so the full-space run alone would stack the deck against DBSCAN. Both results reported; `eps`/`min_samples` re-derived separately per space (k-distance plot and dimensionality both change)

## Evaluation

- [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [Davies-Bouldin](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html), [Calinski-Harabasz](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html) — computed across all three algorithms/tracks as the common cross-algorithm yardstick. Caveat carried into the comparison, not hidden: all three are centroid/distance-based and implicitly reward convex clusters, so they structurally favor K-Means's geometry over DBSCAN's — directionally comparable, not strictly equivalent, across algorithms
- **[DBCV](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)** ([FelSiq/DBCV](https://github.com/FelSiq/DBCV) implementation) — required primary validation metric for DBSCAN specifically, not optional. Scores density-connectedness rather than distance-to-centroid, so it's the correct tool for what DBSCAN actually produces, unlike the three metrics above. Handles noise (`-1`) natively as part of the score — no exclusion needed. Where DBCV and the silhouette/DB/CH verdicts on DBSCAN disagree is reported explicitly, not resolved by picking one
- Qualitative segment profiling — per-cluster RFM/category/review/delivery summary table, PCA 2D scatter
- **Stability check** — for each algorithm/space/track: draw a random subsample (without replacement) of the customer set, re-run that algorithm's full pipeline on it (parameters re-derived fresh, not reused — `k` re-selected via elbow/silhouette, DBSCAN `eps`/`min_samples` re-derived), then compare cluster labels for that subsample's points against their labels from the full-data clustering via [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) (point-to-point correspondence works because it's the same points, not two independent resamples). Repeat n=20 times, report mean ± std ARI per algorithm/space/track. For DBSCAN, noise (`-1`) is kept as its own label in the ARI comparison — noise-vs-noise agreement counts as agreement, noise-vs-cluster counts as disagreement, same as any other label pair.

**DBCV compute constraint.** DBCV builds a mutual-reachability MST over the full pairwise distance graph — effectively O(n²) memory/compute. Feasible directly on the repeat-only track (~3k customers). Not feasible on the full-base track (~96k unique customers) at full size — computed there across 5 independent random subsamples (~5k–10k each), reported as mean ± std, explicitly labeled an estimate rather than the exact full-base DBCV.

**Silhouette/DB/CH point-set alignment.** DBSCAN's `-1` (noise) points can't be scored by these three metrics (they assume every point is cluster-assigned), so excluding noise before computing them is necessary — but excluding noise for DBSCAN alone while K-Means/PCA get scored on their full point set means the algorithms are being compared on different subsets, which isn't a fair comparison. Fix: for a given algorithm/space/track, all three algorithms' silhouette/DB/CH are computed on the **same point set** — the subset DBSCAN did *not* label noise in that run. Rule:
- Noise fraction (`% labeled -1`) reported as its own stat, separately, for full context
- If DBSCAN leaves <2 clusters after removing noise, or noise fraction exceeds ~50%, the aligned metrics are flagged unreliable for that run rather than reported at face value

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
- `prince` — MFA (block weighting) and PCA
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
- Escofier, B. & Pagès, J. (1994), [Multiple Factor Analysis (AFMULT package)](https://three-mode.leidenuniv.nl/pdf/e/escofierpages1994csda.pdf) — block-weighting method used to fix category-block dominance
- [prince — Python MFA/PCA implementation](https://github.com/MaxHalford/prince)
- Ester, Kriegel, Sander, Xu (1996), [original DBSCAN paper](https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695-00/papers/00153.pdf)
- Sander, Ester, Kriegel, Xu (1998), [GDBSCAN — min_samples heuristic](https://static.aminer.org/pdf/PDF/000/307/216/a_density_based_approach_for_clustering_spatial_database.pdf)
- Moulavi, Jaskowiak, Campello, Zimek, Sander (2014), [DBCV — density-based clustering validation](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)
- [FelSiq/DBCV — Python DBCV implementation](https://github.com/FelSiq/DBCV)
- [mlai curriculum](https://github.com/m-adisa/mlai/blob/main/README.md)
