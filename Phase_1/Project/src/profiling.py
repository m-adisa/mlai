"""Cluster profiling and business attribution (theoretical_foundation.md Sec. 6.2).

Describes what each cluster looks like in real units, never in latent
coordinates:

- Route A (Sec. 6.2.2), the reported profile: per cluster and feature, the
  cluster's raw-unit value against the whole population's.
    * continuous features (recency, frequency, monetary, review score,
      delivery delay): Relative Deviation Ratio
      (cluster median - population median) / population IQR. Where the IQR
      is 0 the doc allows a MAD fallback; where the MAD is 0 as well (for
      example frequency on the full base) there is no spread and the
      deviation is left empty.
    * rates (repeat_buyer, multi_category) and category shares: rate
      difference, (cluster rate - population rate), in proportion points.
      repeat_buyer and multi_category are derived here from frequency and the
      share columns; they are the doc's "binary or zero-inflated" case.
- Block attribution (Sec. 6.2.3): BVA, the share of a cluster's deviation
  from the population that comes from each block (rfm / category / extras),
  computed in the weighted space the clustering ran in. The doc says to treat
  it as descriptive (blocks with more dimensions can accumulate more), and to
  confirm it with the Route A features.

Not implemented: Route B (back-projecting centroids). It is the doc's sanity
check, needs more of the fitted transform exposed, and its inverted category
composition is a blurred geometric centre rather than a real basket.

Notes:
- Everything here is association, not causation (Sec. 6.2.4).
- The population baseline is the set of customers on the same side: train
  profiles compare against the train population, held-out profiles against
  the held-out population. Deviations are therefore comparable across sides.
- DBSCAN noise (-1) is profiled as its own group.
- BVA is computed in the space the clustering ran in ("mfa" or "balanced"),
  with the LatentSpace refitted on the train features. Held-out customers go
  through that frozen train transform. BVA values from different spaces are
  not comparable.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.clustering import ALGORITHMS, SPACES
from src.features import BLOCK_NAMES, EXTRAS_COLUMNS, LatentSpace

CONTINUOUS_FEATURES = (
    "recency_days",
    "frequency",
    "monetary_raw",
    "avg_review_score",
    "avg_delivery_delay",
)
RATE_FEATURES = ("repeat_buyer", "multi_category")
SHARE_PREFIX = "category_share__"
SIDES = ("train", "test")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _as_spaces(result: Mapping) -> dict[str, dict]:
    """{space: fitted}. A single cluster_space() result (it has a "Z" key) is
    wrapped and named from its block_balance flag."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be the dict returned by run_clustering() or cluster_space().")
    if "Z" in result:
        balanced = result["latent_diagnostics"]["block_balance"]
        return {"balanced" if balanced else "mfa": result}
    return dict(result)


def _choose(value, allowed, name: str) -> tuple[str, ...]:
    items = (value,) if isinstance(value, str) else tuple(value)
    if not items:
        raise ValueError(f"`{name}` needs at least one of {list(allowed)}.")
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown {name} {unknown}; choose from {list(allowed)}.")
    return items


def _profile_table(features: pd.DataFrame) -> pd.DataFrame:
    """Raw-unit columns used by Route A, indexed by customer_unique_id."""
    indexed = features.set_index("customer_unique_id")
    share_columns = [c for c in indexed.columns if c.startswith(SHARE_PREFIX)]
    table = indexed[list(CONTINUOUS_FEATURES) + share_columns].copy()
    table["repeat_buyer"] = (indexed["frequency"] >= 2).astype(float)
    table["multi_category"] = ((indexed[share_columns] > 0).sum(axis=1) >= 2).astype(float)
    return table


def _weighted_matrix(model: LatentSpace, features: pd.DataFrame) -> pd.DataFrame:
    """The weighted matrix A the clustering ran on (before PCA), for any table,
    using the quantities the LatentSpace stored when it was fitted. Mirrors
    LatentSpace._assemble; swap to a public method if one is added.
    Rows are complete cases, indexed by customer_unique_id."""
    complete = features.dropna(subset=list(EXTRAS_COLUMNS))
    blocks = [complete[columns].to_numpy() for columns in model.block_columns_]
    weighted = [
        scaler.transform(block) / sigma1
        for scaler, block, sigma1 in zip(model.scalers_, blocks, model.sigma1_)
    ]
    if model.block_balance:
        weighted = [w / norm for w, norm in zip(weighted, model.balance_norms_)]
    return pd.DataFrame(np.hstack(weighted), index=complete["customer_unique_id"].to_numpy())


def _tag(frame: pd.DataFrame, space: str, algorithm: str, side: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "side", side)
    frame.insert(0, "algorithm", algorithm)
    frame.insert(0, "space", space)
    return frame


# ---------------------------------------------------------------------------
# ROUTE A, BLOCK ATTRIBUTION, SIZES
# ---------------------------------------------------------------------------


def _sizes(labels: pd.Series) -> pd.DataFrame:
    sizes = labels.value_counts().sort_index().rename_axis("cluster").rename("n_customers").reset_index()
    sizes["share"] = sizes["n_customers"] / len(labels)
    return sizes


def _route_a(table: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Long format: one row per (cluster, feature)."""
    rows = []

    continuous = list(CONTINUOUS_FEATURES)
    population = table[continuous]
    population_median = population.median()
    iqr = population.quantile(0.75) - population.quantile(0.25)
    mad = (population - population_median).abs().median()
    cluster_median = population.groupby(labels).median()
    for cluster in cluster_median.index:
        for feature in continuous:
            if iqr[feature] > 0:
                spread, kind = float(iqr[feature]), "RDR (IQR)"
            elif mad[feature] > 0:
                spread, kind = float(mad[feature]), "RDR (MAD)"
            else:
                spread, kind = np.nan, "none (no spread)"
            cluster_value = float(cluster_median.at[cluster, feature])
            population_value = float(population_median[feature])
            rows.append({
                "cluster": cluster,
                "feature": feature,
                "family": "continuous",
                "stat": "median",
                "population_value": population_value,
                "cluster_value": cluster_value,
                "spread": spread,
                "deviation": (cluster_value - population_value) / spread if not np.isnan(spread) else np.nan,
                "deviation_kind": kind,
            })

    share_columns = [c for c in table.columns if c.startswith(SHARE_PREFIX)]
    rate_columns = list(RATE_FEATURES) + share_columns
    population_rate = table[rate_columns].mean()
    cluster_rate = table[rate_columns].groupby(labels).mean()
    for cluster in cluster_rate.index:
        for column in rate_columns:
            is_share = column.startswith(SHARE_PREFIX)
            cluster_value = float(cluster_rate.at[cluster, column])
            population_value = float(population_rate[column])
            rows.append({
                "cluster": cluster,
                "feature": column[len(SHARE_PREFIX):] if is_share else column,
                "family": "category" if is_share else "rate",
                "stat": "mean share" if is_share else "rate",
                "population_value": population_value,
                "cluster_value": cluster_value,
                "spread": np.nan,
                "deviation": cluster_value - population_value,
                "deviation_kind": "mean share difference" if is_share else "rate difference",
            })
    return pd.DataFrame(rows)


def _block_attribution(A: pd.DataFrame, labels: pd.Series, edges: np.ndarray) -> pd.DataFrame:
    """BVA per cluster from the cluster means of the weighted matrix."""
    squared = A.groupby(labels).mean() ** 2
    parts = {
        name: squared.iloc[:, int(start):int(stop)].sum(axis=1)
        for name, start, stop in zip(BLOCK_NAMES, edges[:-1], edges[1:])
    }
    total = sum(parts.values())
    valid = total > 0

    result = pd.DataFrame({f"bva_{name}": part.where(valid) / total.where(valid) for name, part in parts.items()})
    typical = float(np.sqrt((A ** 2).sum(axis=1).mean()))   # a typical customer's own distance from the centre
    result["distance_from_center"] = np.sqrt(total) / typical

    shares = result[[f"bva_{name}" for name in BLOCK_NAMES]].to_numpy()
    dominant = np.array(BLOCK_NAMES, dtype=object)[np.nan_to_num(shares, nan=-1.0).argmax(axis=1)]
    dominant[~valid.to_numpy()] = None
    result["dominant_block"] = dominant

    result.insert(0, "n_customers", labels.value_counts().reindex(result.index))
    return result.rename_axis("cluster").reset_index()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def profile_clustering(
    features: pd.DataFrame,
    result: Mapping,
    spaces=None,
    algorithms=None,
    test_features: pd.DataFrame | None = None,
    test_labels: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Profile whatever run_clustering() fitted.

    Args:
        features: the train feature table given to run_clustering().
        result: the run_clustering() output (any spaces / algorithms).
        spaces, algorithms: restrict what is profiled (default: all fitted).
        test_features, test_labels: also profile the held-out customers.
            test_labels is evaluate_clustering()'s `scores["test_labels"]`
            (customer_unique_id plus one <space>_<algorithm> column each).
            Pass both or neither.

    Returns {"sizes", "route_a", "block_attribution"}: long-format tables with
    leading columns space, algorithm, side ("train" / "test") and cluster.
    route_a has one row per (cluster, feature) with the raw-unit
    population_value and cluster_value, the spread used, the deviation and
    how it was computed (deviation_kind); family is "continuous", "rate" or
    "category". Deviations of different families are in different units
    (spreads vs proportion points), so compare them only within a family.
    """
    if (test_features is None) != (test_labels is None):
        raise ValueError("Pass test_features and test_labels together, or neither.")

    available = _as_spaces(result)
    chosen_spaces = tuple(available) if spaces is None else _choose(spaces, available, "spaces")
    chosen_algorithms = ALGORITHMS if algorithms is None else _choose(algorithms, ALGORITHMS, "algorithms")

    train_table = _profile_table(features)
    test_table = _profile_table(test_features) if test_features is not None else None

    sizes, route_a, blocks = [], [], []
    for space in chosen_spaces:
        fitted = available[space]
        model = LatentSpace(block_balance=SPACES[space]).fit(features)
        A_train = _weighted_matrix(model, features)
        A_test = _weighted_matrix(model, test_features) if test_features is not None else None

        for algorithm in chosen_algorithms:
            if algorithm not in fitted:
                continue
            sides = [(
                "train", train_table, A_train,
                pd.Series(fitted[algorithm]["labels"], index=fitted["customer_ids"]),
            )]
            column = f"{space}_{algorithm}"
            if test_labels is not None and column in test_labels.columns:
                sides.append((
                    "test", test_table, A_test,
                    pd.Series(test_labels[column].to_numpy(), index=test_labels["customer_unique_id"].to_numpy()),
                ))

            for side, table, A, labels in sides:
                sizes.append(_tag(_sizes(labels), space, algorithm, side))
                route_a.append(_tag(_route_a(table.loc[labels.index], labels), space, algorithm, side))
                blocks.append(_tag(_block_attribution(A.loc[labels.index], labels, model.edges_), space, algorithm, side))

    if not sizes:
        raise ValueError("Nothing to profile: the result has no fitted algorithms for the chosen spaces.")
    return {
        "sizes": pd.concat(sizes, ignore_index=True),
        "route_a": pd.concat(route_a, ignore_index=True),
        "block_attribution": pd.concat(blocks, ignore_index=True),
    }


def top_deviations(
    profile: Mapping[str, pd.DataFrame],
    space: str,
    algorithm: str,
    side: str = "train",
    n: int = 6,
) -> pd.DataFrame:
    """The n largest deviations per cluster, ranked within each family
    (families are in different units, so they are never ranked together)."""
    route_a = _slice(profile["route_a"], space, algorithm, side).dropna(subset=["deviation"]).copy()
    route_a["abs_deviation"] = route_a["deviation"].abs()
    route_a = route_a.sort_values(["cluster", "family", "abs_deviation"], ascending=[True, True, False])
    return route_a.groupby(["cluster", "family"]).head(n).drop(columns="abs_deviation").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


def _slice(table: pd.DataFrame, space: str, algorithm: str, side: str) -> pd.DataFrame:
    selected = table[(table["space"] == space) & (table["algorithm"] == algorithm) & (table["side"] == side)]
    if selected.empty:
        available = table[["space", "algorithm", "side"]].drop_duplicates().values.tolist()
        raise ValueError(f"No profile for ({space!r}, {algorithm!r}, {side!r}); available: {available}.")
    return selected


def _cluster_name(cluster: int) -> str:
    return "noise" if cluster == -1 else f"cluster {cluster}"


def _heatmap_matrix(frame: pd.DataFrame, features: list[str], clusters: list[int]):
    """(deviation, hover text) matrices, rows = features, columns = clusters."""
    deviation = frame.pivot(index="feature", columns="cluster", values="deviation").reindex(index=features, columns=clusters)
    cluster_value = frame.pivot(index="feature", columns="cluster", values="cluster_value").reindex(index=features, columns=clusters)
    population_value = frame.pivot(index="feature", columns="cluster", values="population_value").reindex(index=features, columns=clusters)
    text = [
        [f"cluster {c:.3g} vs population {p:.3g}" for c, p in zip(c_row, p_row)]
        for c_row, p_row in zip(cluster_value.to_numpy(), population_value.to_numpy())
    ]
    return deviation.to_numpy(), text


def plot_profile(
    profile: Mapping[str, pd.DataFrame],
    space: str,
    algorithm: str,
    side: str = "train",
    view: str = "route_a",
    max_categories: int = 12,
) -> go.Figure:
    """Plot one profile (a single space, algorithm and side).

    view="route_a": two heatmaps, features by cluster. Top: continuous
        features as robust deviations (units of the population IQR or MAD).
        Bottom: rates and the `max_categories` categories with the largest
        deviation anywhere, as differences in proportion points. Red = above
        the population, blue = below; hover shows the raw values.
    view="blocks": stacked bars of block attribution (BVA) per cluster, with
        the distance of the cluster mean from the population centre.
    """
    if view not in ("route_a", "blocks"):
        raise ValueError("view must be 'route_a' or 'blocks'.")

    sizes = _slice(profile["sizes"], space, algorithm, side)
    clusters = sizes["cluster"].tolist()
    labels_x = [f"{_cluster_name(c)} ({s:.1%})" for c, s in zip(sizes["cluster"], sizes["share"])]
    title = f"{space} · {algorithm} · {side} profile"

    if view == "blocks":
        blocks = _slice(profile["block_attribution"], space, algorithm, side).set_index("cluster").reindex(clusters)
        labels_with_distance = [
            f"{label}<br>distance {d:.2f}" for label, d in zip(labels_x, blocks["distance_from_center"])
        ]
        figure = go.Figure()
        for name in BLOCK_NAMES:
            figure.add_bar(x=labels_with_distance, y=blocks[f"bva_{name}"], name=name)
        figure.update_layout(
            barmode="stack", title=f"{title}: block attribution", template="plotly_white",
            yaxis_title="share of deviation from the population",
        )
        return figure

    route_a = _slice(profile["route_a"], space, algorithm, side)
    continuous = route_a[route_a["family"] == "continuous"]
    rates = route_a[route_a["family"] == "rate"]
    categories = route_a[route_a["family"] == "category"]

    top_categories = (
        categories.assign(abs_deviation=categories["deviation"].abs())
        .groupby("feature")["abs_deviation"].max()
        .nlargest(max_categories).index.tolist()
    )
    continuous_features = list(CONTINUOUS_FEATURES)
    difference_features = list(RATE_FEATURES) + top_categories
    difference_frame = pd.concat([rates, categories[categories["feature"].isin(top_categories)]])

    z_top, text_top = _heatmap_matrix(continuous, continuous_features, clusters)
    z_bottom, text_bottom = _heatmap_matrix(difference_frame, difference_features, clusters)
    limit_top = float(np.nanmax(np.abs(z_top))) if np.isfinite(z_top).any() else 1.0
    limit_bottom = float(np.nanmax(np.abs(z_bottom))) if np.isfinite(z_bottom).any() else 1.0

    figure = make_subplots(
        rows=2, cols=1, vertical_spacing=0.14,
        row_heights=[len(continuous_features), len(difference_features)],
        subplot_titles=(
            "Spending and experience: median vs population (robust deviation)",
            "Rates and category shares: difference vs population (proportion points)",
        ),
    )
    hover = "%{y}<br>%{x}<br>deviation %{z:.3f}<br>%{text}<extra></extra>"
    figure.add_trace(go.Heatmap(
        z=z_top, x=labels_x, y=continuous_features, text=text_top, coloraxis="coloraxis",
        texttemplate="%{z:.2f}", hovertemplate=hover,
    ), row=1, col=1)
    figure.add_trace(go.Heatmap(
        z=z_bottom, x=labels_x, y=difference_features, text=text_bottom, coloraxis="coloraxis2",
        texttemplate="%{z:.2f}", hovertemplate=hover,
    ), row=2, col=1)
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=60 * (len(continuous_features) + len(difference_features)) + 240,
        coloraxis=dict(colorscale="RdBu_r", cmin=-limit_top, cmax=limit_top,
                       colorbar=dict(title="deviation", len=0.35, y=0.8)),
        coloraxis2=dict(colorscale="RdBu_r", cmin=-limit_bottom, cmax=limit_bottom,
                        colorbar=dict(title="difference", len=0.35, y=0.22)),
    )
    return figure
