"""Clustering for Olist customer segmentation.

This module fits models. It imports `fit_latent_space` and `eps_from_k_distance` from
features.py and does not reach into raw feature columns directly.

Two latent spaces, each clustered by both algorithms (up to four fits):
- "mfa": block standardization + MFA weighting, then global PCA.
- "balanced": the same with block_balance=True (equal total variance per
  block) before global PCA. Metrics from the two spaces are not comparable
  with each other.
Any subset of spaces and algorithms can be chosen per call.

- K-Means: global convex partitioning. K is selected via silhouette within Z.
  Multiple initializations are required because Lloyd's algorithm only finds a
  local optimum.
- DBSCAN: density-based, noise-aware. eps and min_samples come from
  `features.eps_from_k_distance`, derived separately in each space, pass algorithms="kmeans" to skip it.
"""

from __future__ import annotations

from collections.abc import Iterable

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

# Latent-space name -> block_balance flag passed to fit_latent_space.
SPACES = {"mfa": False, "balanced": True}
DEFAULT_SPACES = tuple(SPACES)
ALGORITHMS = ("kmeans", "dbscan")


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


def _select(value, allowed, name: str) -> tuple[str, ...]:
    """Normalize a str or iterable of names to a validated tuple."""
    items = (value,) if isinstance(value, str) else tuple(value)
    if not items:
        raise ValueError(f"`{name}` needs at least one of {list(allowed)}.")
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown {name} {unknown}; choose from {list(allowed)}.")
    return items


def cluster_space(
    features: pd.DataFrame,
    space: str,
    algorithms: str | Iterable[str] = ALGORITHMS,
    k_range: range = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_N_INIT,
    seed: int = DEFAULT_SEED,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> dict:
    """One latent space, the chosen algorithms: build Z once, run the
    K-Means sweep and final fit, derive DBSCAN's (eps, min_samples) from
    that same Z and fit it.

    Chosen k is the silhouette-maximizing k from the sweep.

    Returns:
        Z, customer_ids, latent_diagnostics: fit_latent_space() output.
        kmeans: {k_sweep, chosen_k, labels, model}   (if requested)
        dbscan: {eps, min_samples, labels, model, summary}   (if requested)
    """
    if space not in SPACES:
        raise ValueError(f"Unknown space {space!r}; choose from {list(SPACES)}.")
    algorithms = _select(algorithms, ALGORITHMS, "algorithms")

    Z, customer_ids, latent_diagnostics = fit_latent_space(
        features, block_balance=SPACES[space]
    )
    result = {
        "Z": Z,
        "customer_ids": customer_ids,
        "latent_diagnostics": latent_diagnostics,
    }

    if "kmeans" in algorithms:
        k_sweep = select_k(
            Z, k_range=k_range, n_init=n_init, seed=seed,
            silhouette_sample_size=silhouette_sample_size,
        )
        chosen_k = int(k_sweep.loc[k_sweep["silhouette"].idxmax(), "k"])
        labels, model = fit_kmeans(Z, chosen_k, n_init=n_init, seed=seed)
        result["kmeans"] = {
            "k_sweep": k_sweep,
            "chosen_k": chosen_k,
            "labels": labels,
            "model": model,
        }

    if "dbscan" in algorithms:
        min_samples = MIN_SAMPLES_MULTIPLIER * Z.shape[1]
        eps = eps_from_k_distance(Z, min_samples)
        labels, model = fit_dbscan(Z, eps=eps, min_samples=min_samples)
        result["dbscan"] = {
            "eps": eps,
            "min_samples": min_samples,
            "labels": labels,
            "model": model,
            "summary": dbscan_summary(labels),
        }

    return result


def run_clustering(
    features: pd.DataFrame,
    spaces: str | Iterable[str] = DEFAULT_SPACES,
    algorithms: str | Iterable[str] = ALGORITHMS,
    k_range: range = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_N_INIT,
    seed: int = DEFAULT_SEED,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> dict:
    """Cluster in the chosen latent spaces with the chosen algorithms.

    Examples:
        run_clustering(feats)                                     # 2 K-Means + 2 DBSCAN
        run_clustering(feats, spaces="balanced")                  # choose "mfa" or "balanced" latent space
        run_clustering(feats, algorithms="kmeans")                # choose "kmeans" or "dbscan" algorithm
        run_clustering(feats, spaces="mfa", algorithms="kmeans")  # one fit

    Returns a dict keyed by space name ("mfa", "balanced"), each value
    being cluster_space()'s result, e.g. result["balanced"]["kmeans"]["labels"].
    Arguments are validated before any model is fit.
    """
    spaces = _select(spaces, SPACES, "spaces")
    algorithms = _select(algorithms, ALGORITHMS, "algorithms")

    return {
        space: cluster_space(
            features, space, algorithms=algorithms, k_range=k_range,
            n_init=n_init, seed=seed,
            silhouette_sample_size=silhouette_sample_size,
        )
        for space in spaces
    }


def label_table(result: dict) -> pd.DataFrame:
    """One row per customer, one column per fitted clustering, named
    <space>_<algorithm> (mfa_kmeans, balanced_dbscan, ...). Takes the
    output of run_clustering(). Both spaces use the same complete-case
    customers in the same order, so the columns line up; that is checked.
    """
    table = None
    for space, fitted in result.items():
        ids = fitted["customer_ids"]
        if table is None:
            table = pd.DataFrame({"customer_unique_id": ids})
        elif not np.array_equal(table["customer_unique_id"].to_numpy(), ids):
            raise ValueError(f"customer_ids in space {space!r} differ from the first space.")
        for algorithm in ALGORITHMS:
            if algorithm in fitted:
                table[f"{space}_{algorithm}"] = fitted[algorithm]["labels"]
    return table
