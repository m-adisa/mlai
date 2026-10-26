# Project 1.2: Customer Segmentation

Phase 1 project of the [mlai curriculum](https://github.com/m-adisa/mlai/blob/main/README.md). It segments Olist marketplace customers with K-Means and DBSCAN on a PCA-reduced latent space, then tests whether the structure found is reproducible, generalizes to unseen customers, and means something. This README says how the project is organized and how to run it; it does not repeat the maths or the findings.

## Where things are

| Document | Answers |
|---|---|
| `README.md` (this file) | What is in the project and how to run it |
| [`theoretical_foundation.md`](theoretical_foundation.md) | Why each step is mathematically justified, including block balancing (§3.2.4), held-out evaluation (§5.4) and where the implementation departs from the reference outline (§7.3) |
| [`insights.md`](insights.md) | What the experiment found, how reliable it is, and what it supports |

## Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 100k orders, 2016 to 2018. Tables used: customers, orders, order items, order reviews, products and the category translation. `customer_id` is per order; `customer_unique_id` is the real customer, and everything aggregates on it. Only `delivered` orders are used.

About 97% of customers have exactly one order, so the project runs two tracks through the same pipeline:

- **Full base:** all customers.
- **Repeat customers:** customers with two or more orders (about 3%).

## Pipeline

| Stage | Code | Reference |
|---|---|---|
| Load and validate the nine tables | `data_loading.load_olist` | theory §1 |
| Customer features: RFM, category spend shares, review score, delivery delay | `features.build_train_test_features` | theory §2 |
| Latent space: block standardization, MFA weighting, optional block balancing, PCA | `features.LatentSpace`, `features.preprocessing_diagnostics` | theory §3 |
| Clustering: K-Means (k by silhouette) and DBSCAN (eps by k-distance elbow) | `clustering.run_clustering` | theory §4 |
| Evaluation: metrics, DBCV, subsampling stability against a null baseline, held-out generalization | `evaluation.evaluate_clustering` | theory §5 |
| Profiling: raw-unit cluster profiles and block attribution | `profiling.profile_clustering` | theory §6.2 |
| Plots | `plotting.plot_clustering`, `profiling.plot_profile` | none |

Two latent spaces exist and are never compared with each other: `mfa` (the math doc's baseline) and `balanced` (every block given an equal share of total variance). Feature validation reports both; clustering onward runs on `balanced`, which is selected with `spaces="balanced"` (see `insights.md` for why).

## Settings

| Setting | Value | Where |
|---|---|---|
| Category coverage target | 0.87 of spend (22 categories plus `other`) | `features.DEFAULT_TARGET_SPEND_COVERAGE` |
| PCA variance target / cap | 0.87 / 26 (cap never binds) | `features.DEFAULT_VAR_TARGET`, `DEFAULT_D_MAX` |
| DBSCAN `min_samples` | 2 x latent dimensions | `features.MIN_SAMPLES_MULTIPLIER` |
| Train/test split | 45% test, by customer, stratified on the repeat flag, seed 42 | `features.TEST_SIZE`, `SPLIT_RANDOM_STATE` |
| DBCV | 5 subsamples of 5,000 points | `evaluation.DEFAULT_DBCV_*` |
| Stability | 80% subsamples; 7 pairs (full base), 9 (repeat); the doc recommends 20 | notebook constants |

The coverage target, the variance target and the dimension cap were calibrated together. Re-derive all three if one changes.

## Running it

Open `notebook.ipynb` and run it top to bottom. The first run downloads the dataset through `kagglehub` into `data/raw/` (gitignored).

```python
from src.data_loading import load_olist, validate_tables
from src.features import build_train_test_features, validate_features, preprocessing_diagnostics
from src.clustering import run_clustering
from src.evaluation import evaluate_clustering
from src.profiling import profile_clustering, top_deviations, plot_profile
from src.plotting import plot_clustering

tables = load_olist()
train_feats, test_feats = build_train_test_features(tables)
validate_features(train_feats, train_feats.attrs["category_columns"])
diagnostics = preprocessing_diagnostics(train_feats, test_feats)        # both spaces, train vs test

result = run_clustering(train_feats, spaces="balanced")                 # K-Means and DBSCAN
scores = evaluate_clustering(result, train_feats, test_features=test_feats)
profile = profile_clustering(train_feats, result, test_features=test_feats, test_labels=scores["test_labels"])
```

Every stage accepts `spaces=` and `algorithms=` to run a subset. DBSCAN is the memory-hungry step on the full base; pass `algorithms="kmeans"` if it fails. Repeat customers: `features.repeat_customers_only(train_feats)`, run through the same calls.

## Structure

```
Phase_1/Project/
├── README.md
├── theoretical_foundation.md
├── insights.md
├── notebook.ipynb
├── data/raw/                # gitignored, downloaded on first run
├── images                   # images from the notebook
└── src/
    ├── data_loading.py      # download, load, validate
    ├── features.py          # features, split, LatentSpace, diagnostics
    ├── clustering.py        # K-Means, DBSCAN, k selection
    ├── evaluation.py        # metrics, DBCV, stability, generalization
    ├── profiling.py         # Route A profiles, block attribution, profile plots
    └── plotting.py          # cluster scatter, sizes, k sweep
```

## Environment

Dependencies are in the `Pipfile`. `dbcv` is installed from GitHub (`pip install "git+https://github.com/FelSiq/DBCV"`). With SciPy 1.18 or later, `dbcv` needs the integer-rank shim already in `evaluation.py`.

## Resources

- [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Escofier & Pagès (1994), [Multiple Factor Analysis](https://three-mode.leidenuniv.nl/pdf/e/escofierpages1994csda.pdf)
- Westerhuis, Kourti & MacGregor (1998), [Analysis of multiblock and hierarchical PCA and PLS models](https://literature.learnche.org/item/95/analysis-of-multiblock-and-hierarchical-pca-and-pls-models)
- Ester et al. (1996), [DBSCAN](https://webdocs.cs.ualberta.ca/~zaiane/courses/cmput695-00/papers/00153.pdf); Sander et al. (1998), [GDBSCAN](https://static.aminer.org/pdf/PDF/000/307/216/a_density_based_approach_for_clustering_spatial_database.pdf)
- Moulavi et al. (2014), [DBCV](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf); [FelSiq/DBCV](https://github.com/FelSiq/DBCV)
- scikit-learn: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
