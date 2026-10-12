"""Interactive plots for Olist clustering results (plotly).

Single entry point: `plot_clustering(result, view=...)`. `result` is whatever
clustering.run_clustering() returned (any choice of spaces / algorithms), or a
single clustering.cluster_space() result. Whatever was fitted is plotted; what
was not fitted is skipped.

Views:
- "scatter": one panel per (space, algorithm) in Z, coloured by cluster, noise
  in gray. Rows are spaces, columns are algorithms. 2D (PC1/PC2) or 3D (PC1-3);
  hover shows customer_unique_id; click legend entries to hide/show clusters.
- "sizes": cluster-size bars per panel (shows degenerate or lopsided results).
- "k_sweep": K-Means inertia (elbow) and silhouette per k, chosen k marked.

Scatter plots show only 2-3 of the Z dimensions. Axis titles give each
component's share of the total variance, so you can see how much of the space
the picture covers. Overlap on the plot does not imply overlap in Z.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

ALGORITHM_NAMES = {"kmeans": "K-Means", "dbscan": "DBSCAN"}
VIEWS = ("scatter", "sizes", "k_sweep")
PALETTE = qualitative.Dark24
NOISE_COLOR = "lightgray"
DEFAULT_MAX_POINTS = 20_000
ROW_HEIGHT = 450


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


def _algorithms_present(spaces: dict[str, dict]) -> list[str]:
    return [a for a in ALGORITHM_NAMES if any(a in fitted for fitted in spaces.values())]


def _panel_title(space: str, algorithm: str, fitted: dict) -> str:
    if algorithm not in fitted:
        return ""
    if algorithm == "kmeans":
        return f"{space} · K-Means (k={fitted['kmeans']['chosen_k']})"
    s = fitted["dbscan"]["summary"]
    n = s["n_clusters"]
    return f"{space} · DBSCAN ({n} cluster{'' if n == 1 else 's'}, {s['noise_fraction']:.1%} noise)"


def _sample(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))


def _pc_labels(Z: np.ndarray, realized_variance: float) -> list[str]:
    """Axis titles with each component's share of the TOTAL variance
    (Z's component variances are PCA eigenvalues; realized_variance is the
    fraction of total variance Z keeps)."""
    var = Z.var(axis=0)
    share = var / var.sum() * realized_variance
    return [f"PC{i + 1} ({s:.1%} of variance)" for i, s in enumerate(share)]


def _label_name(label: int) -> str:
    return "noise" if label == -1 else f"cluster {label}"


def _label_color(label: int) -> str:
    return NOISE_COLOR if label == -1 else PALETTE[int(label) % len(PALETTE)]


def _finish(fig: go.Figure, rows: int, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        height=ROW_HEIGHT * rows + 80,
        template="plotly_white",
        legend=dict(groupclick="toggleitem"),
    )
    return fig


# ---------------------------------------------------------------------------
# VIEWS
# ---------------------------------------------------------------------------


def _scatter_view(spaces: dict[str, dict], dims: int, max_points: int, seed: int) -> go.Figure:
    algorithms = _algorithms_present(spaces)
    rows, cols = len(spaces), len(algorithms)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene" if dims == 3 else "xy"}] * cols for _ in range(rows)],
        subplot_titles=[
            _panel_title(space, algo, fitted)
            for space, fitted in spaces.items()
            for algo in algorithms
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.12 if dims == 2 else 0.06,
    )

    for r, (space, fitted) in enumerate(spaces.items(), start=1):
        Z, ids = fitted["Z"], fitted["customer_ids"]
        if Z.shape[1] < dims:
            raise ValueError(f"Space {space!r} has only {Z.shape[1]} dimensions; cannot plot {dims}D.")
        idx = _sample(len(Z), max_points, seed)
        z, sampled_ids = Z[idx], ids[idx]
        axis_titles = _pc_labels(Z, fitted["latent_diagnostics"]["realized_variance"])

        for c, algo in enumerate(algorithms, start=1):
            if algo not in fitted:
                continue
            labels = fitted[algo]["labels"][idx]
            group = f"{space}-{algo}"
            group_title = _panel_title(space, algo, fitted)

            for label in np.unique(labels):
                m = labels == label
                name = _label_name(int(label))
                marker = dict(
                    size=3 if dims == 2 else 2,
                    color=_label_color(int(label)),
                    opacity=0.5 if label == -1 else 0.75,
                )
                common = dict(
                    mode="markers",
                    name=name,
                    legendgroup=group,
                    legendgrouptitle_text=group_title,
                    customdata=sampled_ids[m],
                    hovertemplate=f"%{{customdata}}<br>{name}<extra></extra>",
                    marker=marker,
                )
                if dims == 2:
                    trace = go.Scattergl(x=z[m, 0], y=z[m, 1], **common)
                else:
                    trace = go.Scatter3d(x=z[m, 0], y=z[m, 1], z=z[m, 2], **common)
                fig.add_trace(trace, row=r, col=c)

            if dims == 2:
                fig.update_xaxes(title_text=axis_titles[0], row=r, col=c)
                fig.update_yaxes(title_text=axis_titles[1], row=r, col=c)
            else:
                fig.update_scenes(
                    xaxis_title=axis_titles[0],
                    yaxis_title=axis_titles[1],
                    zaxis_title=axis_titles[2],
                    row=r,
                    col=c,
                )

    shown = min(max_points, max(len(f["Z"]) for f in spaces.values()))
    return _finish(fig, rows, f"Clusters in latent space Z ({dims}D, {shown:,} points sampled per panel)")


def _sizes_view(spaces: dict[str, dict], log_y: bool) -> go.Figure:
    algorithms = _algorithms_present(spaces)
    rows, cols = len(spaces), len(algorithms)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            _panel_title(space, algo, fitted)
            for space, fitted in spaces.items()
            for algo in algorithms
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
    )

    for r, (space, fitted) in enumerate(spaces.items(), start=1):
        for c, algo in enumerate(algorithms, start=1):
            if algo not in fitted:
                continue
            values, counts = np.unique(fitted[algo]["labels"], return_counts=True)
            fig.add_trace(
                go.Bar(
                    x=[_label_name(int(v)) for v in values],
                    y=counts,
                    marker_color=[_label_color(int(v)) for v in values],
                    text=[f"{n:,}" for n in counts],
                    textposition="outside",
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            fig.update_yaxes(title_text="customers", type="log" if log_y else "linear", row=r, col=c)

    return _finish(fig, rows, "Cluster sizes")


def _k_sweep_view(spaces: dict[str, dict]) -> go.Figure:
    with_kmeans = {s: f for s, f in spaces.items() if "kmeans" in f}
    if not with_kmeans:
        raise ValueError("view='k_sweep' needs K-Means results; none were fitted.")
    rows = len(with_kmeans)
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[
            f"{space} · {metric}"
            for space in with_kmeans
            for metric in ("inertia (elbow)", "silhouette")
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.14,
    )

    for r, (space, fitted) in enumerate(with_kmeans.items(), start=1):
        sweep = fitted["kmeans"]["k_sweep"]
        chosen = fitted["kmeans"]["chosen_k"]
        for c, metric in enumerate(("inertia", "silhouette"), start=1):
            fig.add_trace(
                go.Scatter(x=sweep["k"], y=sweep[metric], mode="lines+markers", showlegend=False),
                row=r,
                col=c,
            )
            fig.add_vline(
                x=chosen,
                line_dash="dash",
                line_color="crimson",
                annotation_text=f"chosen k={chosen}",
                row=r,
                col=c,
            )
            fig.update_xaxes(title_text="k", row=r, col=c)

    return _finish(fig, rows, "K-Means k sweep")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def plot_clustering(
    result: Mapping,
    view: str = "scatter",
    dims: int = 2,
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    log_y: bool = False,
) -> go.Figure:
    """Plot whatever run_clustering() (or cluster_space()) produced.

    Args:
        result: output of run_clustering() with any spaces/algorithms.
        view: "scatter" (default), "sizes", or "k_sweep".
        dims: 2 or 3 -- scatter view only.
        max_points: random sample per panel, scatter view only (the same
            customers are drawn in every panel of a space).
        seed: sampling seed.
        log_y: log-scale y axis -- sizes view only.

    Returns the plotly Figure; as the last expression of a notebook cell it
    renders inline, or call .show() / .write_html("clusters.html").
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; choose from {list(VIEWS)}.")
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3.")

    spaces = _as_spaces(result)
    if not spaces:
        raise ValueError("result is empty -- nothing to plot.")

    if view == "scatter":
        return _scatter_view(spaces, dims, max_points, seed)
    if view == "sizes":
        return _sizes_view(spaces, log_y)
    return _k_sweep_view(spaces)
