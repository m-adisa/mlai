"""Clustering for Olist customer segmentation.

This module fits models. It imports `fit_latent_space` and `eps_from_k_distance` from
features.py and does not reach into raw feature columns directly.

- K-Means: global convex partitioning. K is selected via silhouette within Z.
  Multiple initializations are required because Lloyd's algorithm only finds a
  local optimum.
- DBSCAN: density-based, noise-aware. eps and min_samples come from
  `features.eps_from_k_distance`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.features import MIN_SAMPLES_MULTIPLIER, eps_from_k_distance, fit_latent_space

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_K_RANGE = range(2, 16)
DEFAULT_N_INIT = 10
DEFAULT_SEED = 0
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000


# ---------------------------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------------------------


def select_k(
    Z: np.ndarray,
    k_range: range = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_N_INIT,
    seed: int = DEFAULT_SEED,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Fit K-Means for each k in `k_range`, report inertia (for an elbow
    plot) and silhouette, both within Z.

    Every k gets its own `n_init` restarts and a fixed `seed`, so the
    sweep is reproducible.

    Returns a DataFrame with columns [k, inertia, silhouette], one row
    per k -- plot inertia for the elbow, read off silhouette's max for
    the criterion the doc names, and compare the two before committing
    to a k.
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)
        sil = silhouette_score(
            Z,
            model.labels_,
            sample_size=silhouette_sample_size,
            random_state=seed,
        )
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(
    Z: np.ndarray,
    k: int,
    n_init: int = DEFAULT_N_INIT,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, KMeans]:
    """Fit the final K-Means model at a chosen k. Every point gets a
    cluster label -- K-Means has no noise concept.
    """
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)
    return model.labels_, model


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


def fit_dbscan(Z: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, DBSCAN]:
    """Fit DBSCAN at a given (eps, min_samples).

    Returns labels where -1 marks noise. Noise is not dropped, imputed, or relabeled here.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
    return model.labels_, model


def dbscan_summary(labels: np.ndarray) -> dict:
    """Descriptive counts only -- not a validation metric. How many
    clusters DBSCAN found and what fraction of points it called noise.
    """
    unique = set(labels.tolist())
    n_clusters = len(unique - {-1})
    n_noise = int((labels == -1).sum())
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / len(labels),
        "n_points": len(labels),
    }


# ---------------------------------------------------------------------------
# LABEL ASSEMBLY
# ---------------------------------------------------------------------------


def label_customers(
    customer_ids: np.ndarray, labels: np.ndarray, column_name: str
) -> pd.DataFrame:
    """Tidy customer_unique_id -> cluster-label table, for merging back
    onto the feature table for profiling. One row per customer, in the
    same order fit_latent_space() returned customer_ids in -- caller is
    responsible for passing labels that came from clustering that same
    Z/customer_ids pair.
    """
    return pd.DataFrame({"customer_unique_id": customer_ids, column_name: labels})


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------


def run_clustering(
    features: pd.DataFrame,
    k_range: range = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_N_INIT,
    seed: int = DEFAULT_SEED,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> dict:
    """End-to-end: build Z once, run the K-Means sweep and final fit,
    derive DBSCAN's (eps, min_samples) from that same Z and fit it.

    Chosen k is the silhouette-maximizing k from the sweep.

    Returns:
        Z, customer_ids, latent_diagnostics: fit_latent_space() output.
        kmeans: {k_sweep, chosen_k, labels, model}
        dbscan: {eps, min_samples, labels, model, summary}
    """
    Z, customer_ids, latent_diagnostics = fit_latent_space(features)

    k_sweep = select_k(
        Z, k_range=k_range, n_init=n_init, seed=seed,
        silhouette_sample_size=silhouette_sample_size,
    )
    chosen_k = int(k_sweep.loc[k_sweep["silhouette"].idxmax(), "k"])
    kmeans_labels, kmeans_model = fit_kmeans(Z, chosen_k, n_init=n_init, seed=seed)

    min_samples = MIN_SAMPLES_MULTIPLIER * Z.shape[1]
    eps = eps_from_k_distance(Z, min_samples)
    dbscan_labels, dbscan_model = fit_dbscan(Z, eps=eps, min_samples=min_samples)

    return {
        "Z": Z,
        "customer_ids": customer_ids,
        "latent_diagnostics": latent_diagnostics,
        "kmeans": {
            "k_sweep": k_sweep,
            "chosen_k": chosen_k,
            "labels": kmeans_labels,
            "model": kmeans_model,
        },
        "dbscan": {
            "eps": eps,
            "min_samples": min_samples,
            "labels": dbscan_labels,
            "model": dbscan_model,
            "summary": dbscan_summary(dbscan_labels),
        },
    }
