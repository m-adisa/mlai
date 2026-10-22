"""Evaluation for Olist clustering results.

Takes what clustering.run_clustering() returned and measures it. Four
independent checks, any of which can be run on its own:

- internal_metrics: silhouette, Davies-Bouldin, Calinski-Harabasz
  (theoretical_foundation.md Sec. 5.1).
- dbcv_scores: Density-Based Clustering Validation (Sec. 5.2), the metric
  designed for what DBSCAN produces.
- stability: paired-subsample ARI with a full pipeline re-fit (Sec. 5.3),
  plus a structure-free null baseline (Sec. 5.3.4).
- generalization: does the structure fitted on TRAIN customers hold for
  held-out TEST customers? Two parts: the latent space (how much of the
  test customers' variance the train PCA captures, per block) and the
  clusters (test customers assigned to the train clusters, then scored and
  compared with train).

Same-space rule (Sec. 5.1.2, 6.1): Silhouette, DB, CH and DBCV are all
distance-based, so values from different latent spaces ("mfa" vs
"balanced") must not be ranked against each other. Compare algorithms
inside one space. ARI stability is representation-agnostic but only says
the partition is reproducible, not that it is useful (Sec. 6.1).

Train/test protocol: `result` and `features` are always the TRAIN side
(what run_clustering() was fitted on). Test customers are only ever
projected with the train-fitted latent space and assigned with the
train-fitted clusters; nothing is refitted on them (the one exception, the
K-Means "refit ARI", is a yardstick and never labels a customer).

Where this module goes beyond or adapts the doc:
- DBCV runs on rescaled Z. The package returns an exact 0.0 when Z's
  axes are tiny (std about 0.004-0.015 here) although DBCV is scale
  invariant in theory, so Z is divided by one global constant first.
  This does not change the geometry.
- DBCV uses Euclidean distance, as the doc's formulas do. The package
  default is squared Euclidean (the original MATLAB code).
- DBCV on the full base is an estimate from random subsamples (Sec. 5.2.3
  requires subsampling: it is O(N^2)).
- DBCV is called through a SciPy compatibility shim: SciPy >= 1.18 returns
  float64 from rankdata(method="dense"), and dbcv uses those ranks as array
  indices (IndexError). See _integer_ranks.
- Stability aligns the two subsample labelings on customer_unique_id,
  because fit_latent_space drops incomplete customers inside each
  subsample, so array position is not a stable key.
- Generalization does not report Calinski-Harabasz: it grows with the
  number of points, so a smaller test set would look worse for no reason.
  Silhouette and Davies-Bouldin are size-invariant.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping

import dbcv as dbcv_package
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from src.clustering import (
    ALGORITHMS,
    DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    SPACES,
    fit_dbscan,
    fit_kmeans,
)
from src.features import (
    MIN_SAMPLES_MULTIPLIER,
    LatentSpace,
    eps_from_k_distance,
    fit_latent_space,
)

EVALUATIONS = ("metrics", "dbcv", "stability", "generalization")
# What evaluate_clustering() runs when `run` is not given; "generalization"
# is added to it only when test_features is passed.
DEFAULT_EVALUATIONS = ("metrics", "dbcv", "stability")

HIGH_NOISE_FRACTION = 0.5

DEFAULT_DBCV_SAMPLE_SIZE = 5_000
DEFAULT_DBCV_N_SUBSAMPLES = 5
DBCV_JITTER = 1e-3  # on unit-scale Z; only used when exact duplicates block DBCV

DEFAULT_N_REPEATS = 20
DEFAULT_SUBSAMPLE_FRAC = 0.80
STABLE_MEAN_ARI = 0.75
STABLE_SD_ARI = 0.05

# Rule of thumb, not a test: below this many test points per cluster the
# held-out numbers are noisy (the repeat-purchase track is the usual case).
MIN_TEST_POINTS_PER_CLUSTER = 30

# Columns fit_latent_space reads (category CLR columns are added by prefix).
# Needed to build the structure-free null baseline.
_LATENT_COLUMNS = (
    "recency_days",
    "frequency",
    "monetary",
    "avg_review_score",
    "avg_delivery_delay",
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _as_spaces(result: Mapping) -> dict[str, dict]:
    """Normalize to {space_name: fitted}. A single cluster_space() result
    (it has a "Z" key) is wrapped and named from its block_balance flag."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be the dict returned by run_clustering() or cluster_space().")
    if "Z" in result:
        balanced = result["latent_diagnostics"]["block_balance"]
        return {"balanced" if balanced else "mfa": result}
    return dict(result)


def _names(value, allowed, name: str) -> tuple[str, ...]:
    items = (value,) if isinstance(value, str) else tuple(value)
    if not items:
        raise ValueError(f"`{name}` needs at least one of {list(allowed)}.")
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown {name} {unknown}; choose from {list(allowed)}.")
    return items


def _filter(result: Mapping, spaces, algorithms) -> dict[str, dict]:
    """Restrict a clustering result to the chosen spaces / algorithms."""
    available = _as_spaces(result)
    chosen_spaces = list(available) if spaces is None else _names(spaces, available, "spaces")
    chosen_algorithms = ALGORITHMS if algorithms is None else _names(algorithms, ALGORITHMS, "algorithms")
    return {
        space: {
            key: value
            for key, value in available[space].items()
            if key not in ALGORITHMS or key in chosen_algorithms
        }
        for space in chosen_spaces
    }


def _algorithms_in(fitted: dict) -> list[str]:
    return [a for a in ALGORITHMS if a in fitted]


# ---------------------------------------------------------------------------
# INTERNAL METRICS: SILHOUETTE / DAVIES-BOULDIN / CALINSKI-HARABASZ
# ---------------------------------------------------------------------------


def _distance_metrics(Z: np.ndarray, labels: np.ndarray, sample_size: int | None, seed: int) -> tuple:
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters > len(labels) - 1:
        return np.nan, np.nan, np.nan
    subsample = sample_size if sample_size is not None and len(Z) > sample_size else None
    try:
        silhouette = float(silhouette_score(Z, labels, sample_size=subsample, random_state=seed))
    except ValueError:  # a subsample that happened to contain a single cluster
        silhouette = np.nan
    return (
        silhouette,
        float(davies_bouldin_score(Z, labels)),
        float(calinski_harabasz_score(Z, labels)),
    )


def _metric_row(space, algorithm, basis, Z, labels, keep, sample_size, seed) -> dict:
    excluded = 1.0 - float(keep.mean())
    z, lab = Z[keep], labels[keep]
    silhouette, davies_bouldin, calinski_harabasz = _distance_metrics(z, lab, sample_size, seed)

    notes = []
    n_clusters = len(np.unique(lab))
    if n_clusters < 2:
        notes.append("fewer than 2 clusters: metrics undefined")
    if excluded > HIGH_NOISE_FRACTION:
        notes.append("over 50% of points excluded: unreliable")
    return {
        "space": space,
        "algorithm": algorithm,
        "basis": basis,
        "n_points": int(keep.sum()),
        "n_clusters": n_clusters,
        "excluded_fraction": excluded,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "note": "; ".join(notes),
    }


def internal_metrics(
    result: Mapping,
    spaces=None,
    algorithms=None,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    seed: int = 0,
) -> pd.DataFrame:
    """Silhouette (higher is better), Davies-Bouldin (lower), Calinski-Harabasz
    (higher), per space and algorithm.

    Rows per space:
    - K-Means, "all points".
    - DBSCAN, "non-noise points" (these metrics cannot score noise).
    - K-Means, "DBSCAN non-noise points": K-Means restricted to the same
      points DBSCAN kept, so the K-Means and DBSCAN rows are scored on an
      identical point set. Only when both algorithms were fitted.

    `excluded_fraction` is the share of all points left out of scoring.
    Silhouette is O(n^2), so it is computed on a random sample of
    `silhouette_sample_size` points when there are more (None = never).
    Compare rows only within a space.

    Also accepts the output of assign_test_clusters(), which scores the
    held-out customers on their own.
    """
    rows = []
    for space, fitted in _filter(result, spaces, algorithms).items():
        Z = fitted["Z"]
        everyone = np.ones(len(Z), dtype=bool)
        kmeans, dbscan = fitted.get("kmeans"), fitted.get("dbscan")

        if kmeans is not None:
            rows.append(_metric_row(space, "kmeans", "all points", Z, kmeans["labels"], everyone,
                                    silhouette_sample_size, seed))
        if dbscan is not None:
            non_noise = dbscan["labels"] != -1
            rows.append(_metric_row(space, "dbscan", "non-noise points", Z, dbscan["labels"], non_noise,
                                    silhouette_sample_size, seed))
            if kmeans is not None:
                rows.append(_metric_row(space, "kmeans", "DBSCAN non-noise points", Z, kmeans["labels"],
                                        non_noise, silhouette_sample_size, seed))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DBCV
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _integer_ranks():
    """Make scipy.stats.rankdata return integers for its integer methods.

    dbcv relabels clusters with `rankdata(y, method="dense") - 1` and then
    indexes arrays with the result. SciPy >= 1.18 returns float64 there, so
    dbcv fails with "IndexError: only integers, slices ... are valid
    indices". Dense/min/max/ordinal ranks are integers by definition, so
    they are cast back for the duration of the call and scipy is restored
    afterwards. A no-op on SciPy versions that already return integers.
    """
    original = scipy.stats.rankdata

    def rankdata(a, method="average", *args, **kwargs):
        ranks = original(a, method, *args, **kwargs)
        if method in ("min", "max", "dense", "ordinal") and np.issubdtype(ranks.dtype, np.floating):
            return ranks.astype(np.int64)
        return ranks

    scipy.stats.rankdata = rankdata
    try:
        yield
    finally:
        scipy.stats.rankdata = original


def _dbcv_euclidean(X: np.ndarray, y: np.ndarray) -> float:
    with _integer_ranks():
        return dbcv_package.dbcv(X, y, metric="euclidean")


def _dbcv_once(X: np.ndarray, y: np.ndarray, jitter_rng: np.random.Generator) -> tuple[float, bool]:
    """DBCV on one set of points; returns (value, jittered). NaN when fewer
    than two non-noise clusters of size >= 2 exist (the doc requires K >= 2;
    the package would return a meaningless value for a single cluster)."""
    _, counts = np.unique(y[y != -1], return_counts=True)
    if (counts >= 2).sum() < 2:
        return np.nan, False
    try:
        return float(_dbcv_euclidean(X, y)), False
    except ValueError as error:
        if "Duplicated" not in str(error):
            raise
    # Exact duplicate customers make the core distance undefined; a tiny
    # jitter (1e-3 of the unit scale) separates them without moving structure.
    jittered = X + jitter_rng.normal(0.0, DBCV_JITTER, X.shape)
    return float(_dbcv_euclidean(jittered, y)), True


def dbcv_scores(
    result: Mapping,
    spaces=None,
    algorithms=None,
    sample_size: int = DEFAULT_DBCV_SAMPLE_SIZE,
    n_subsamples: int = DEFAULT_DBCV_N_SUBSAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """DBCV per space and algorithm (higher is better, range -1 to 1).

    DBCV counts noise in N, so a partition that labels many points as noise
    is penalized. It needs at least 2 non-noise clusters; otherwise the
    value is NaN. It is O(n^2): populations up to `sample_size` are scored
    exactly, larger ones as the mean and std of `n_subsamples` random
    subsamples of `sample_size` points (an estimate, not the exact
    full-population DBCV; noise stays in each subsample at its natural rate).

    DBCV is distance-based: compare rows only within a space.

    Also accepts the output of assign_test_clusters(), which scores the
    held-out customers on their own.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for space, fitted in _filter(result, spaces, algorithms).items():
        Z = fitted["Z"]
        X_all = Z / Z.std(axis=0).mean()  # one global constant; geometry unchanged
        for algorithm in _algorithms_in(fitted):
            labels = fitted[algorithm]["labels"]
            if len(Z) <= sample_size:
                draws, scope = [np.arange(len(Z))], "full population"
            else:
                draws = [np.sort(rng.choice(len(Z), sample_size, replace=False)) for _ in range(n_subsamples)]
                scope = f"estimate from {n_subsamples} subsamples of {sample_size:,} points"

            values, jittered = [], 0
            for idx in draws:
                value, was_jittered = _dbcv_once(X_all[idx], labels[idx], rng)
                values.append(value)
                jittered += was_jittered
            valid = np.array([v for v in values if not np.isnan(v)])

            notes = [scope]
            if len(valid) < len(values):
                notes.append(f"{len(values) - len(valid)} subsample(s) had fewer than 2 clusters")
            if jittered:
                notes.append(f"duplicates jittered in {jittered} run(s)")
            rows.append({
                "space": space,
                "algorithm": algorithm,
                "dbcv_mean": float(valid.mean()) if len(valid) else np.nan,
                "dbcv_std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                "n_valid": len(valid),
                "note": "; ".join(notes),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STABILITY
# ---------------------------------------------------------------------------


def _null_features(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Structure-free reference (Sec. 5.3.4): same marginals, every latent
    column permuted independently, which destroys all dependence."""
    rng = np.random.default_rng(seed)
    columns = list(_LATENT_COLUMNS) + [c for c in features.columns if c.startswith("category_clr__")]
    null = features.copy()
    for column in columns:
        null[column] = rng.permutation(features[column].to_numpy())
    return null


def _fit_subsample(sub: pd.DataFrame, space: str, algorithms: list[str], k: int | None, seed: int) -> dict:
    """Re-fit the whole pipeline on one subsample: scaler, MFA sigma_1, PCA,
    d*, and for DBSCAN eps. K-Means keeps the k chosen on the full data
    (Sec. 7.1 `stability`); everything data-derived is re-derived."""
    Z, ids, _ = fit_latent_space(sub, block_balance=SPACES[space])
    fits = {}
    if "kmeans" in algorithms:
        labels, _ = fit_kmeans(Z, k, seed=seed)
        fits["kmeans"] = (ids, labels)
    if "dbscan" in algorithms:
        min_samples = MIN_SAMPLES_MULTIPLIER * Z.shape[1]
        eps = eps_from_k_distance(Z, min_samples)
        labels, _ = fit_dbscan(Z, eps=eps, min_samples=min_samples)
        fits["dbscan"] = (ids, labels)
    return fits


def _stability_run(features, targets, n_repeats, frac, seed, verbose, tag) -> dict:
    """B pairs of independent subsamples; ARI on the overlap, aligned on
    customer_unique_id. targets = {space: {"algorithms": [...], "k": int|None}}."""
    rng = np.random.default_rng(seed)
    n = len(features)
    m = int(frac * n)
    stats = {
        (space, algorithm): {"ari": [], "noise": [], "clusters": [], "overlap": []}
        for space, target in targets.items()
        for algorithm in target["algorithms"]
    }

    for b in range(n_repeats):
        if verbose:
            print(f"[stability {tag}] pair {b + 1}/{n_repeats}", flush=True)
        sub_a = features.iloc[np.sort(rng.choice(n, m, replace=False))]
        sub_b = features.iloc[np.sort(rng.choice(n, m, replace=False))]
        for space, target in targets.items():
            fits_a = _fit_subsample(sub_a, space, target["algorithms"], target["k"], seed)
            fits_b = _fit_subsample(sub_b, space, target["algorithms"], target["k"], seed + 1)
            for algorithm in target["algorithms"]:
                ids_a, labels_a = fits_a[algorithm]
                ids_b, labels_b = fits_b[algorithm]
                common, pos_a, pos_b = np.intersect1d(ids_a, ids_b, assume_unique=True, return_indices=True)
                record = stats[(space, algorithm)]
                record["ari"].append(adjusted_rand_score(labels_a[pos_a], labels_b[pos_b]))  # -1 kept as a label
                record["overlap"].append(len(common))
                record["clusters"].append(np.mean([len(set(labels_a.tolist()) - {-1}),
                                                   len(set(labels_b.tolist()) - {-1})]))
                if algorithm == "dbscan":
                    record["noise"].append(np.mean([(labels_a == -1).mean(), (labels_b == -1).mean()]))
    return stats


def _sd(values) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def stability(
    features: pd.DataFrame,
    result: Mapping,
    spaces=None,
    algorithms=None,
    n_repeats: int = DEFAULT_N_REPEATS,
    subsample_frac: float = DEFAULT_SUBSAMPLE_FRAC,
    null_baseline: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Subsampling stability (Sec. 5.3): `n_repeats` pairs of independent
    80% subsamples, the entire pipeline re-fitted on each, ARI on the
    customers both subsamples contain. DBSCAN noise (-1) is kept as a label.

    Args:
        features: the SAME feature table that was given to run_clustering()
            (the train table; full base or repeat-only), not the latent space.
        result: the run_clustering() output; supplies which spaces and
            algorithms exist and the K-Means k chosen on the full data.
        null_baseline: also run the identical procedure on a structure-free
            copy (columns permuted independently). A stability number is
            only meaningful against it. Doubles the runtime.

    Returns one row per (space, algorithm): ari_mean, ari_std (sample std),
    mean clusters (excluding noise), mean noise fraction (DBSCAN), mean
    overlap, the null baseline's ari_mean / ari_std and the margin over it,
    and `meets_heuristic`
    (mean >= 0.75 and std <= 0.05: conventional heuristics, not tests; the
    std understates uncertainty because the pairs share data). Per-pair
    ARIs are in `.attrs["ari_by_pair"]`.

    Cost: each pair re-fits Z for every space, and DBSCAN re-derives eps.
    On the full base this is long and DBSCAN is memory-hungry; restrict
    `spaces` / `algorithms` or lower `n_repeats` to fit your machine.
    A DBSCAN row with fewer than 2 clusters is flagged: its ARI is not
    informative (identical single-cluster partitions score 1).
    """
    chosen = _filter(result, spaces, algorithms)
    targets = {}
    for space, fitted in chosen.items():
        algos = _algorithms_in(fitted)
        if algos:
            targets[space] = {
                "algorithms": algos,
                "k": fitted["kmeans"]["chosen_k"] if "kmeans" in fitted else None,
            }
    if not targets:
        raise ValueError("Nothing to evaluate: the result has no fitted algorithms for the chosen spaces.")

    real = _stability_run(features, targets, n_repeats, subsample_frac, seed, verbose, "data")
    null = (
        _stability_run(_null_features(features, seed + 10_000), targets, n_repeats, subsample_frac,
                       seed, verbose, "null baseline")
        if null_baseline
        else None
    )

    rows, per_pair = [], {}
    for (space, algorithm), record in real.items():
        ari = record["ari"]
        mean_ari, sd_ari = float(np.mean(ari)), _sd(ari)
        mean_clusters = float(np.mean(record["clusters"]))
        row = {
            "space": space,
            "algorithm": algorithm,
            "n_pairs": len(ari),
            "ari_mean": mean_ari,
            "ari_std": sd_ari,
            "mean_clusters": mean_clusters,
            "mean_noise_fraction": float(np.mean(record["noise"])) if record["noise"] else np.nan,
            "mean_overlap": float(np.mean(record["overlap"])),
            "meets_heuristic": bool(mean_ari >= STABLE_MEAN_ARI and sd_ari <= STABLE_SD_ARI),
        }
        notes = []
        if null is not None:
            null_ari = null[(space, algorithm)]["ari"]
            row["null_ari_mean"] = float(np.mean(null_ari))
            row["null_ari_std"] = _sd(null_ari)
            row["ari_minus_null"] = mean_ari - row["null_ari_mean"]
            if mean_ari <= row["null_ari_mean"]:
                notes.append("at or below the null baseline: not evidence of structure")
        if mean_clusters < 2:
            notes.append("fewer than 2 clusters: ARI not informative")
        row["note"] = "; ".join(notes)
        rows.append(row)
        per_pair[f"{space}/{algorithm}"] = [float(a) for a in ari]

    table = pd.DataFrame(rows)
    table.attrs["ari_by_pair"] = per_pair
    return table


# ---------------------------------------------------------------------------
# GENERALIZATION: TRAIN-FITTED LATENT SPACE AND CLUSTERS ON HELD-OUT CUSTOMERS
# ---------------------------------------------------------------------------


def assign_kmeans(model: KMeans, Z_test: np.ndarray) -> np.ndarray:
    """Nearest train centroid."""
    return model.predict(Z_test)


def assign_dbscan(model: DBSCAN, Z_test: np.ndarray) -> np.ndarray:
    """Label of the nearest TRAIN core point within model.eps, else -1.

    DBSCAN has no predict: its clusters are the connected components of core
    points found in the data it was fitted on, so there is no model to apply
    to a new point. This is DBSCAN's own border-point rule applied
    inductively. Test points cannot become cores or merge clusters, so the
    train clustering stays fixed, and eps / min_samples (derived on train)
    are used as they are.
    """
    labels = np.full(len(Z_test), -1, dtype=int)
    if len(model.core_sample_indices_) == 0:
        return labels
    core_labels = model.labels_[model.core_sample_indices_]
    distance, index = NearestNeighbors(n_neighbors=1).fit(model.components_).kneighbors(Z_test)
    inside = distance[:, 0] <= model.eps  # DBSCAN's neighborhood is closed: d <= eps
    labels[inside] = core_labels[index[inside, 0]]
    return labels


def _rebuild_latent(train_features: pd.DataFrame, space: str, fitted: dict) -> LatentSpace:
    """Refit the train latent space (deterministic) and check it reproduces
    the Z in `result`, so `result` is guaranteed to come from
    `train_features` and the projection uses the very same scalers / PCA."""
    latent = LatentSpace(block_balance=SPACES[space]).fit(train_features)
    same = (
        latent.Z_.shape == fitted["Z"].shape
        and np.array_equal(latent.customer_ids_, fitted["customer_ids"])
        and np.allclose(latent.Z_, fitted["Z"])
    )
    if not same:
        raise ValueError(
            f"result[{space!r}] was not produced from the `features` given "
            "(the rebuilt train latent space does not match its Z)."
        )
    return latent


def assign_test_clusters(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    result: Mapping,
    spaces=None,
    algorithms=None,
) -> dict[str, dict]:
    """Project test customers into each train latent space and assign them
    to the train clusters. Nothing is refitted.

    Returns a run_clustering()-shaped dict, {space: {Z, customer_ids,
    latent_diagnostics, kmeans: {labels}, dbscan: {labels}}}, so
    clustering.label_table(), internal_metrics() and dbcv_scores() all work
    on it directly. latent_diagnostics is LatentSpace.diagnose(): train vs.
    test numbers for the held-out latent space.
    """
    held_out = {}
    for space, fitted in _filter(result, spaces, algorithms).items():
        latent = _rebuild_latent(train_features, space, fitted)
        Z_test, customer_ids = latent.transform(test_features)
        entry = {
            "Z": Z_test,
            "customer_ids": customer_ids,
            "latent_diagnostics": {"block_balance": latent.block_balance, **latent.diagnose(test_features)},
        }
        if "kmeans" in fitted:
            entry["kmeans"] = {"labels": assign_kmeans(fitted["kmeans"]["model"], Z_test)}
        if "dbscan" in fitted:
            entry["dbscan"] = {"labels": assign_dbscan(fitted["dbscan"]["model"], Z_test)}
        held_out[space] = entry
    return held_out


def _proportions(train_labels: np.ndarray, test_labels: np.ndarray) -> pd.DataFrame:
    """Share of customers per cluster label, train vs. test (-1 = noise)."""
    train = pd.Series(train_labels).value_counts(normalize=True).rename("train_share")
    test = pd.Series(test_labels).value_counts(normalize=True).rename("test_share")
    table = pd.concat([train, test], axis=1).fillna(0.0).sort_index()
    table.index.name = "label"
    return table


def latent_generalization(
    result: Mapping,
    test_result: Mapping,
    spaces=None,
) -> pd.DataFrame:
    """Held-out latent-space quality, train vs. test, per space.

    The train PCA (d_star components) is applied to the test customers.
    Rows per space: "overall", then one per block (rfm, category, extras).

    - fraction_kept_*: share of the variance that survives the PCA cut. For
      "overall" this is the realized variance; for a block it is the share
      of that block's own variance kept in Z. A test value well below the
      train value means the train axes do not describe new customers.
    - share_in_Z_*: each block's share of the variance kept in Z.
    - reconstruction_error_ratio (overall only): per-customer squared
      reconstruction error on test over train; 1.0 = same, above 1 = worse.

    Compare train and test within one space; spaces are not comparable.
    """
    rows = []
    for space, fitted in _filter(result, spaces, None).items():
        diag = test_result[space]["latent_diagnostics"]
        block_variance = diag["block_variance"]
        base = {
            "space": space,
            "n_train": len(fitted["Z"]),
            "n_test": diag["n_customers_used"],
            "n_test_excluded_missing_extras": diag["n_customers_excluded_missing_extras"],
        }
        realized = diag["realized_variance"]
        rows.append({
            **base,
            "scope": "overall",
            "fraction_kept_train": realized["train"],
            "fraction_kept_test": realized["test"],
            "fraction_kept_gap": realized["train"] - realized["test"],
            "share_in_Z_train": np.nan,
            "share_in_Z_test": np.nan,
            "reconstruction_error_ratio": diag["reconstruction_error_ratio"],
        })
        for block, kept in block_variance["fraction_kept_in_Z"].items():
            share = block_variance["share_in_Z"][block]
            rows.append({
                **base,
                "scope": block,
                "fraction_kept_train": kept["train"],
                "fraction_kept_test": kept["test"],
                "fraction_kept_gap": kept["train"] - kept["test"],
                "share_in_Z_train": share["train"],
                "share_in_Z_test": share["test"],
                "reconstruction_error_ratio": np.nan,
            })
    return pd.DataFrame(rows)


def cluster_generalization(
    result: Mapping,
    test_result: Mapping,
    spaces=None,
    algorithms=None,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster generalization, train vs. test, per space and algorithm.

    Returns (table, proportions):

    table, one row per (space, algorithm); `basis` is "all points" for
    K-Means and "non-noise points" for DBSCAN (as in internal_metrics):
    - n_train / n_test: points scored; n_clusters_*: non-noise clusters.
    - noise_fraction_*: DBSCAN only.
    - silhouette_* / davies_bouldin_*: same sampling and seed on both sides;
      silhouette_gap = train - test (positive = test is worse).
    - size_tv_distance: total-variation distance between the train and test
      cluster-size distributions (0 = identical, 1 = disjoint; noise counts
      as its own category); max_abs_size_diff: its largest single cluster.
    - train_clusters_empty_in_test: train clusters no test customer reached.
    - inertia_per_point_ratio (K-Means): test over train mean squared
      distance to the assigned centroid; 1.0 = same.
    - refit_ari (K-Means): ARI between the train-model assignments of the
      test customers and an independent K-Means fitted on test at the same
      k. High = the train partition is what test would find on its own.
      Not computed for DBSCAN: its eps depends on sample size, so an
      independent test-only DBSCAN differs for reasons unrelated to
      generalization.

    proportions: long-format (space, algorithm, label, train_share,
    test_share), the data behind the size-shift columns.

    Calinski-Harabasz is deliberately omitted: it grows with n, so a smaller
    test set would look worse without being worse.
    """
    rows, proportion_tables = [], []
    for space, fitted in _filter(result, spaces, algorithms).items():
        held = test_result[space]
        Z_train, Z_test = fitted["Z"], held["Z"]
        for algorithm in _algorithms_in(fitted):
            if algorithm not in held:
                continue
            labels_train = fitted[algorithm]["labels"]
            labels_test = held[algorithm]["labels"]
            is_dbscan = algorithm == "dbscan"
            keep_train = labels_train != -1 if is_dbscan else np.ones(len(labels_train), dtype=bool)
            keep_test = labels_test != -1 if is_dbscan else np.ones(len(labels_test), dtype=bool)

            sil_train, db_train, _ = _distance_metrics(
                Z_train[keep_train], labels_train[keep_train], silhouette_sample_size, seed)
            sil_test, db_test, _ = _distance_metrics(
                Z_test[keep_test], labels_test[keep_test], silhouette_sample_size, seed)
            n_clusters_train = len(np.unique(labels_train[keep_train]))
            n_clusters_test = len(np.unique(labels_test[keep_test]))

            table = _proportions(labels_train, labels_test)
            diff = (table["train_share"] - table["test_share"]).abs()
            empty = int(((table["train_share"] > 0) & (table["test_share"] == 0)).sum())

            inertia_ratio, refit_ari = np.nan, np.nan
            if algorithm == "kmeans":
                model = fitted["kmeans"]["model"]
                train_inertia = model.inertia_ / len(Z_train)
                test_inertia = float((model.transform(Z_test).min(axis=1) ** 2).sum() / len(Z_test))
                inertia_ratio = test_inertia / train_inertia
                # Yardstick only: never used to label a customer.
                refit_labels, _ = fit_kmeans(
                    Z_test, fitted["kmeans"]["chosen_k"], n_init=model.n_init, seed=model.random_state
                )
                refit_ari = float(adjusted_rand_score(labels_test, refit_labels))

            notes = []
            if min(n_clusters_train, n_clusters_test) < 2:
                notes.append("fewer than 2 clusters on one side: silhouette / Davies-Bouldin undefined")
            if empty:
                notes.append(f"{empty} train cluster(s) have no test customers")
            if int(keep_test.sum()) < MIN_TEST_POINTS_PER_CLUSTER * max(n_clusters_test, 1):
                notes.append(f"fewer than {MIN_TEST_POINTS_PER_CLUSTER} test points per cluster: noisy")

            rows.append({
                "space": space,
                "algorithm": algorithm,
                "basis": "non-noise points" if is_dbscan else "all points",
                "n_train": int(keep_train.sum()),
                "n_test": int(keep_test.sum()),
                "n_clusters_train": n_clusters_train,
                "n_clusters_test": n_clusters_test,
                "noise_fraction_train": float((labels_train == -1).mean()) if is_dbscan else np.nan,
                "noise_fraction_test": float((labels_test == -1).mean()) if is_dbscan else np.nan,
                "silhouette_train": sil_train,
                "silhouette_test": sil_test,
                "silhouette_gap": sil_train - sil_test,
                "davies_bouldin_train": db_train,
                "davies_bouldin_test": db_test,
                "size_tv_distance": float(0.5 * diff.sum()),
                "max_abs_size_diff": float(diff.max()),
                "train_clusters_empty_in_test": empty,
                "inertia_per_point_ratio": inertia_ratio,
                "refit_ari": refit_ari,
                "note": "; ".join(notes),
            })
            proportion_tables.append(table.reset_index().assign(space=space, algorithm=algorithm))

    proportions = (
        pd.concat(proportion_tables, ignore_index=True)[["space", "algorithm", "label", "train_share", "test_share"]]
        if proportion_tables
        else pd.DataFrame(columns=["space", "algorithm", "label", "train_share", "test_share"])
    )
    return pd.DataFrame(rows), proportions


def held_out_label_table(test_result: Mapping) -> pd.DataFrame:
    """One row per held-out customer, one column per assignment, named
    <space>_<algorithm> (same layout as clustering.label_table)."""
    table = None
    for space, held in test_result.items():
        ids = held["customer_ids"]
        if table is None:
            table = pd.DataFrame({"customer_unique_id": ids})
        elif not np.array_equal(table["customer_unique_id"].to_numpy(), ids):
            raise ValueError(f"customer_ids in space {space!r} differ from the first space.")
        for algorithm in ALGORITHMS:
            if algorithm in held:
                table[f"{space}_{algorithm}"] = held[algorithm]["labels"]
    return table


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def evaluate_clustering(
    result: Mapping,
    features: pd.DataFrame | None = None,
    run: str | Iterable[str] | None = None,
    spaces=None,
    algorithms=None,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    dbcv_sample_size: int = DEFAULT_DBCV_SAMPLE_SIZE,
    dbcv_n_subsamples: int = DEFAULT_DBCV_N_SUBSAMPLES,
    n_repeats: int = DEFAULT_N_REPEATS,
    subsample_frac: float = DEFAULT_SUBSAMPLE_FRAC,
    null_baseline: bool = True,
    seed: int = 0,
    verbose: bool = True,
    *,
    test_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Run any of the four evaluations on whatever run_clustering() fitted.

    `result` and `features` are the TRAIN side. `run=None` runs "metrics",
    "dbcv" and "stability", plus "generalization" when `test_features` is
    given.

    Examples:
        evaluate_clustering(result, feats)                           # metrics, dbcv, stability
        evaluate_clustering(result, run="metrics")                   # fast, no features needed
        evaluate_clustering(result, run=["metrics", "dbcv"])
        evaluate_clustering(result, feats, run="stability", spaces="balanced", algorithms="kmeans",
                            n_repeats=5, null_baseline=False)
        evaluate_clustering(result, train_feats, run="generalization", test_features=test_feats)

    `features` is required for "stability" (it re-fits on subsamples) and
    for "generalization" (the train latent space is rebuilt from it);
    `test_features` is required for "generalization".

    Returns a dict of DataFrames for what was run:
        "metrics", "dbcv", "stability"
        "generalization"        cluster generalization, one row per space/algorithm
        "latent_generalization" held-out latent space, per space and block
        "cluster_proportions"   cluster-size shares, train vs. test
        "test_labels"           held-out customers' cluster assignments
    Compare rows only within a space (same-space rule).
    """
    if run is None:
        run = DEFAULT_EVALUATIONS + (("generalization",) if test_features is not None else ())
    selected = _names(run, EVALUATIONS, "run")
    if "stability" in selected and features is None:
        raise ValueError("`features` is required for run='stability'.")
    if "generalization" in selected and (features is None or test_features is None):
        raise ValueError("`features` (train) and `test_features` are required for run='generalization'.")

    output = {}
    if "metrics" in selected:
        output["metrics"] = internal_metrics(
            result, spaces, algorithms, silhouette_sample_size=silhouette_sample_size, seed=seed
        )
    if "dbcv" in selected:
        output["dbcv"] = dbcv_scores(
            result, spaces, algorithms, sample_size=dbcv_sample_size,
            n_subsamples=dbcv_n_subsamples, seed=seed,
        )
    if "stability" in selected:
        output["stability"] = stability(
            features, result, spaces, algorithms, n_repeats=n_repeats,
            subsample_frac=subsample_frac, null_baseline=null_baseline, seed=seed, verbose=verbose,
        )
    if "generalization" in selected:
        held_out = assign_test_clusters(features, test_features, result, spaces, algorithms)
        output["generalization"], output["cluster_proportions"] = cluster_generalization(
            result, held_out, spaces, algorithms,
            silhouette_sample_size=silhouette_sample_size, seed=seed,
        )
        output["latent_generalization"] = latent_generalization(result, held_out, spaces)
        output["test_labels"] = held_out_label_table(held_out)
    return output
