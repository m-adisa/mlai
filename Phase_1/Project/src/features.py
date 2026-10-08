"""Per-customer feature engineering for Olist customer segmentation.

This module is responsible for:
- aggregating order-level data to the `customer_unique_id` grain,
- computing RFM, category-preference, and marketplace-extras features,
- applying the feature-level transforms (log1p on monetary, CLR on
  category shares),
- building the single MFA-weighted, PCA-reduced latent space Z that
  the clustering algorithms run in, and the diagnostic numbers that
  parameterize it (PCA component count, MFA block eigenvalues, DBSCAN
  eps, the category-coverage threshold, and a monetary skew check),
- reporting which customers are excluded at each stage,
  and validating the mathematical invariants those transforms depend on.

Order population: only `order_status == 'delivered'` orders are used throughout. All order-level
aggregations join at a safe grain (item -> product -> category, item ->
order -> customer) using pandas `merge(..., validate=...)`, which
hard-fails on any join that would multiply rows -- this is what rules
out the row-multiplication bug documented in theoretical_foundation.md
Sec. 1.2, structurally, not just by convention.

Usage:

    feats = build_customer_features(tables)
    validate_features(feats, feats.attrs["category_columns"])

    diagnostics = preprocessing_diagnostics(feats)   # d*, MFA eigenvalues,
                                                      # DBSCAN eps (single
                                                      # space Z), monetary skew

`build_customer_features` reports population changes via warnings and
via `feats.attrs["population_report"]`; it does not raise on its own
population decisions (those are expected, documented exclusions, not
errors). `validate_features` raises on invariant violations -- those
would mean something is actually broken.

NaN handling note: `fit_latent_space` (and therefore everything in
`preprocessing_diagnostics` that depends on it) requires a dense numeric
matrix, so it runs on the COMPLETE-CASE subset of customers (non-null
avg_review_score and avg_delivery_delay). This module does not impute
those columns for the full population -- that decision is still open
and belongs wherever the final clustering input is assembled. The
diagnostic numbers (d*, MFA weights, eps) are reported together with
how many customers they were computed on, so that scope is visible
rather than implicit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DELIVERED_STATUS = "delivered"
OTHER_CATEGORY_LABEL = "other"

# Category selection: minimum number of categories needed to cover this
# fraction of total spend (README-sourced, not theoretical_foundation.md --
# see module docstring). Replaces a fixed top-N -- the number of
# categories is now a computed quantity, re-derivable if the data changes,
# rather than a guessed constant.
#
# 0.90, not the originally-discussed 0.95: at 0.95 coverage the category
# block needed 33 categories, pushing d90 (components for 90% variance) to
# 34 -- far above any reasonable d_max, so the PCA cap bound hard and
# realized variance was only ~24%. At 0.90 coverage, 25 categories are
# enough and d90 drops to 26, which is achievable without the cap
# destroying fidelity (see DEFAULT_D_MAX below). This pair of constants
# was calibrated together, against the real data -- not chosen
# independently.
DEFAULT_TARGET_SPEND_COVERAGE = 0.90

# Multiplicative-replacement delta for CLR zero handling. Default matches
# theoretical_foundation.md Sec. 2.2.3 / Sec. 7.1 (delta = 1 / C^2, where C
# is the number of category columns including "other"). Exposed as a
# parameter everywhere so the delta-sensitivity sweep the doc recommends
# can be run from the notebook without editing this module.
DEFAULT_DELTA = None  # resolved to 1 / C**2 per-call, C = n_category_cols

# PCA / MFA defaults. var_target=0.90 is theoretical_foundation.md's own
# heuristic (Sec. 3.3.1/7.1 -- "a default to be calibrated, not a
# theorem"). d_max=26 is NOT that doc's default (it proposes 8) -- 26 is
# this project's calibrated value, set to equal d90 at
# DEFAULT_TARGET_SPEND_COVERAGE=0.90, confirmed empirically against the
# real data (see fit_latent_space diagnostics), so that the cap does not
# bind and realized variance actually reaches var_target. If
# DEFAULT_TARGET_SPEND_COVERAGE changes, d90 changes, and this needs
# re-deriving -- it is not independent of that constant.
DEFAULT_D_MAX = 26
DEFAULT_VAR_TARGET = 0.90

# DBSCAN min_samples heuristic (Ester et al., 1996 / Sander et al., 1998),
# applied once, in the single latent space Z that both algorithms cluster
# in (theoretical_foundation.md Sec. 3.3.2/4/5.1.2 -- "always within the
# same latent space Z").
MIN_SAMPLES_MULTIPLIER = 2


# ---------------------------------------------------------------------------
# ORDER-LEVEL PREPARATION
# ---------------------------------------------------------------------------


def valid_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per delivered order, keyed by customer_unique_id.

    Joins orders -> customers (order_id's customer_id is 1:1 with both
    tables, per data_loading.py's identity-semantics validation, so this
    join cannot duplicate rows -- enforced here via validate="many_to_one",
    which would raise if that assumption ever stopped holding). Filters to
    `order_status == 'delivered'` (README "Data filtering rules").

    Columns: order_id, customer_unique_id, order_purchase_timestamp,
    order_estimated_delivery_date, order_delivered_customer_date,
    delivery_delay_days (NaN where order_delivered_customer_date is null
    -- not imputed, per README).
    """
    orders = tables["orders"]
    customers = tables["customers"]

    merged = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    assert len(merged) == len(orders), "unexpected row multiplication joining customers"

    delivered = merged[merged["order_status"] == DELIVERED_STATUS].copy()

    delivered["delivery_delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.days
    # Rows with a null actual delivery date naturally produce NaN above
    # (not imputed); they are dropped only when averaging delay, in
    # compute_extras(), not here -- they still count toward frequency
    # and monetary.

    return delivered[
        [
            "order_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
            "delivery_delay_days",
        ]
    ]


def order_monetary(order_items: pd.DataFrame) -> pd.DataFrame:
    """Per-order monetary total: sum(price) + sum(freight_value).

    Grouping by order_id before anything else is what keeps this safe --
    order_items is one row per item, so this is a plain many-to-one
    reduction, not a join that can multiply rows.
    """
    agg = order_items.groupby("order_id", as_index=False).agg(
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
    )
    agg["monetary_total"] = agg["price_total"] + agg["freight_total"]
    return agg[["order_id", "price_total", "freight_total", "monetary_total"]]


def order_category_spend(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    category_translation: pd.DataFrame,
) -> pd.DataFrame:
    """Per (order_id, category_english) item-price spend.

    Category spend uses `price` only, not freight -- freight is a shipping
    cost, not a category-attributable amount (the RFM monetary feature
    uses price + freight; this is a deliberate, narrower basis).

    Missing or untranslated product_category_name values are mapped to
    OTHER_CATEGORY_LABEL rather than dropped, so their spend still counts
    toward a customer's total (and therefore their category shares still
    sum to 1).

    Join chain: order_items (many) -> products (one, on product_id) ->
    category_translation (one, on product_category_name). Both joins are
    declared many_to_one, so pandas raises if either relationship is ever
    violated, rather than silently duplicating order_items rows.
    """
    merged = order_items.merge(
        products[["product_id", "product_category_name"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    ).merge(
        category_translation,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )
    assert len(merged) == len(order_items), "unexpected row multiplication joining categories"

    merged["category_english"] = merged["product_category_name_english"].fillna(
        OTHER_CATEGORY_LABEL
    )

    return merged.groupby(["order_id", "category_english"], as_index=False).agg(
        category_spend=("price", "sum")
    )


def deduplicated_order_reviews(order_reviews: pd.DataFrame) -> pd.DataFrame:
    """One review score per order_id: latest review_answer_timestamp wins.

    Olist's order_reviews table is not 1-row-per-order: some orders
    received multiple distinct reviews over time (see the dedup
    investigation in this project's history). The composite key
    (review_id, order_id) is already clean -- this function collapses a
    different grain, order_id, which is what feature aggregation needs.

    Rule: for each order_id, keep the review with the latest
    review_answer_timestamp (the customer's final recorded sentiment).
    Ties broken by review_creation_date, then by review_id for a fully
    deterministic result.
    """
    ordered = order_reviews.sort_values(
        ["order_id", "review_answer_timestamp", "review_creation_date", "review_id"]
    )
    return ordered.groupby("order_id", as_index=False).tail(1)[
        ["order_id", "review_score"]
    ]


# ---------------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------------


def compute_rfm(
    orders: pd.DataFrame,
    monetary: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-customer Recency, Frequency, Monetary.

    Semantics, stated explicitly: Frequency and Monetary are computed
    only over delivered orders that have at least one matching
    order_items row (i.e. recoverable monetary data). An order with no
    order_items match is excluded from both frequency and monetary for
    that customer -- it does not reduce the customer's Recency, since
    Recency only needs a purchase timestamp, not monetary data. The
    count of excluded orders is reported via warning (diagnostic, not a
    population decision made silently).

    Args:
        orders: output of valid_orders().
        monetary: output of order_monetary().
        reference_date: anchor for recency. Defaults to the max
            order_purchase_timestamp across `orders` (README: "relative
            to dataset max date").

    Returns columns: customer_unique_id, recency_days, frequency,
    monetary_raw, monetary (log1p of monetary_raw).
    """
    if reference_date is None:
        reference_date = orders["order_purchase_timestamp"].max()

    with_monetary = orders.merge(monetary, on="order_id", how="left", validate="one_to_one")
    assert len(with_monetary) == len(orders), "unexpected row multiplication joining monetary"

    missing_monetary = int(with_monetary["monetary_total"].isna().sum())
    if missing_monetary:
        warnings.warn(
            f"{missing_monetary:,} delivered orders have no matching "
            "order_items rows and were excluded from frequency/monetary "
            "(not from recency -- see compute_rfm docstring).",
            stacklevel=2,
        )
        with_monetary = with_monetary.dropna(subset=["monetary_total"])

    rfm = with_monetary.groupby("customer_unique_id", as_index=False).agg(
        recency_days=("order_purchase_timestamp", lambda s: (reference_date - s.max()).days),
        frequency=("order_id", "count"),
        monetary_raw=("monetary_total", "sum"),
    )
    rfm["monetary"] = np.log1p(rfm["monetary_raw"])
    return rfm


def monetary_skew_report(rfm: pd.DataFrame) -> dict:
    """Diagnostic: does log1p actually tame monetary_raw's skew?

    Not a theoretical_foundation.md section on its own, but a direct
    consequence of its Sec. 2.1 argument (log1p compresses high-value
    outliers) -- that argument predicts an effect, it does not confirm
    the effect is large enough on THIS data. This computes both skews
    so that's checked rather than assumed. Informational only; does not
    raise or alter any feature.
    """
    raw_skew = float(skew(rfm["monetary_raw"]))
    log1p_skew = float(skew(rfm["monetary"]))
    return {
        "raw_skew": raw_skew,
        "log1p_skew": log1p_skew,
        "skew_reduction": raw_skew - log1p_skew,
        "monetary_raw_max_over_p75": float(
            rfm["monetary_raw"].max() / rfm["monetary_raw"].quantile(0.75)
        ),
    }


# ---------------------------------------------------------------------------
# CATEGORY PREFERENCE
# ---------------------------------------------------------------------------


def select_categories_for_coverage(
    category_spend: pd.DataFrame,
    orders: pd.DataFrame,
    target_coverage: float = DEFAULT_TARGET_SPEND_COVERAGE,
) -> tuple[list[str], dict]:
    """Smallest set of categories (by spend, descending) whose cumulative
    spend reaches `target_coverage` of total spend, restricted to
    delivered orders.

    README-sourced default (0.90), not theoretical_foundation.md -- the
    recommendation being implemented here is: pick the category count
    that achieves the target, then hold it fixed for the rest of the
    project, rather than guessing a round number (the project's original
    top-10 default was exactly such a guess). 0.90 rather than the
    originally-discussed 0.95 because of its knock-on effect on PCA
    fidelity -- see DEFAULT_TARGET_SPEND_COVERAGE's comment.

    OTHER_CATEGORY_LABEL is excluded from the candidate ranking -- it's
    a catch-all, not a category to select for coverage.

    Returns (categories, diagnostics) where diagnostics reports the
    achieved coverage (>= target_coverage, since coverage only increases
    in discrete category-sized steps) and the category count.
    """
    in_scope = category_spend[category_spend["order_id"].isin(orders["order_id"])]
    ranked = (
        in_scope[in_scope["category_english"] != OTHER_CATEGORY_LABEL]
        .groupby("category_english")["category_spend"]
        .sum()
        .sort_values(ascending=False)
    )
    total_spend = ranked.sum()
    cumulative_share = ranked.cumsum() / total_spend

    n_categories = int(np.searchsorted(cumulative_share.to_numpy(), target_coverage) + 1)
    n_categories = min(n_categories, len(ranked))

    categories = ranked.index[:n_categories].tolist()
    achieved_coverage = float(cumulative_share.iloc[n_categories - 1])

    diagnostics = {
        "target_coverage": target_coverage,
        "n_categories": n_categories,
        "achieved_coverage": achieved_coverage,
        "n_categories_total": len(ranked),
    }
    return categories, diagnostics


def multiplicative_replacement(
    shares: np.ndarray, delta: float | None = None
) -> np.ndarray:
    """Replace zero shares with delta, rescaling non-zeros so rows still
    sum to 1. Verbatim per theoretical_foundation.md Sec. 2.2.3 / 7.1
    (ground truth for this transform).

    Rows that are entirely NaN (undefined composition -- see
    compute_category_shares) pass through as NaN; this function does not
    decide whether such rows belong in the final population, it only
    avoids crashing on them.
    """
    shares = np.asarray(shares, dtype=float)
    delta = 1.0 / shares.shape[1] ** 2 if delta is None else delta
    zero = shares == 0
    n_zero = zero.sum(axis=1, keepdims=True)
    return np.where(zero, delta, (1.0 - n_zero * delta) * shares)


def clr(shares: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform. Verbatim per
    theoretical_foundation.md Sec. 2.2.2 / 7.1.
    """
    log_shares = np.log(shares)
    return log_shares - log_shares.mean(axis=1, keepdims=True)


def compute_category_shares(
    category_spend: pd.DataFrame,
    orders: pd.DataFrame,
    selected_categories: list[str],
    delta: float | None = None,
) -> pd.DataFrame:
    """Per-customer category spend-share vector (raw shares + CLR columns).

    Categories outside `selected_categories` are folded into
    OTHER_CATEGORY_LABEL, so every customer's share vector has
    len(selected_categories) + 1 dimensions and sums to 1 -- EXCEPT
    customers with zero total category spend (e.g. all-zero-price
    items), for whom a share vector is mathematically undefined (0/0).
    This function does NOT drop those customers; it reports their count
    via warning and returns them with NaN in every share/CLR column. The
    decision of whether they belong in the final clustering population
    is made by the caller (build_customer_features), not here -- feature
    construction and population filtering are kept separate.
    """
    with_customer = category_spend.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    bucketed = with_customer.copy()
    bucketed["category_bucket"] = bucketed["category_english"].where(
        bucketed["category_english"].isin(selected_categories), OTHER_CATEGORY_LABEL
    )

    pivot = (
        bucketed.groupby(["customer_unique_id", "category_bucket"])["category_spend"]
        .sum()
        .unstack(fill_value=0.0)
    )

    category_cols = selected_categories + [OTHER_CATEGORY_LABEL]
    for col in category_cols:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[category_cols]

    row_totals = pivot.sum(axis=1)
    zero_spend_customers = int((row_totals == 0).sum())
    if zero_spend_customers:
        warnings.warn(
            f"{zero_spend_customers:,} customers have zero total category "
            "spend (likely all-zero-price items) -- their category share "
            "is mathematically undefined (0/0) and left as NaN here. "
            "build_customer_features() decides whether to exclude them.",
            stacklevel=2,
        )

    # row_totals == 0 produces NaN via true division, which is the
    # correct representation of "undefined" -- not silently replaced.
    shares = pivot.div(row_totals.replace(0, np.nan), axis=0)

    replaced = multiplicative_replacement(shares.to_numpy(), delta=delta)
    clr_values = clr(replaced)

    raw_cols = {f"category_share__{c}": shares[c].to_numpy() for c in category_cols}
    clr_cols = {
        f"category_clr__{c}": clr_values[:, i] for i, c in enumerate(category_cols)
    }

    result = pd.DataFrame(
        {"customer_unique_id": shares.index, **raw_cols, **clr_cols}
    ).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# MARKETPLACE EXTRAS
# ---------------------------------------------------------------------------


def compute_extras(
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Per-customer avg review score and avg delivery delay.

    Both are left as NaN (not imputed) for customers with no eligible
    orders -- zero-review customers get no review score, and customers
    whose every order has a null delivered_customer_date get no delay
    (README "Data filtering rules"). This missingness is preserved in
    the returned feature table; fit_latent_space() below restricts
    itself to complete cases rather than imputing silently (see module
    docstring) -- the decision of how the full population's missingness
    should ultimately be handled stays open and visible, not made here.
    """
    with_reviews = orders.merge(reviews, on="order_id", how="left", validate="one_to_one")
    assert len(with_reviews) == len(orders), "unexpected row multiplication joining reviews"

    review_avg = with_reviews.groupby("customer_unique_id")["review_score"].mean()

    # delivery_delay_days is already NaN where the actual delivery date is
    # null (set in valid_orders()); .mean() skips NaN by default, so a
    # customer whose every order has a null delivery date gets NaN here,
    # not an imputed value.
    delay_avg = orders.groupby("customer_unique_id")["delivery_delay_days"].mean()

    extras = pd.DataFrame(
        {
            "avg_review_score": review_avg,
            "avg_delivery_delay": delay_avg,
        }
    ).reset_index()

    return extras


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT -- FEATURE TABLE
# ---------------------------------------------------------------------------


def build_customer_features(
    tables: dict[str, pd.DataFrame],
    target_spend_coverage: float = DEFAULT_TARGET_SPEND_COVERAGE,
    delta: float | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the full per-customer feature table.

    Returns one row per customer_unique_id with both raw (interpretable,
    for Route-A profiling per theoretical_foundation.md Sec. 6.2.2) and
    feature-level-transformed columns:

        customer_unique_id
        recency_days, frequency, monetary_raw, monetary            (RFM; monetary = log1p(monetary_raw); see compute_rfm for exact semantics)
        category_share__<cat> ... category_share__other            (raw shares, sum to 1; categories chosen to cover target_spend_coverage of total spend)
        category_clr__<cat> ... category_clr__other                (CLR-transformed, model-ready)
        avg_review_score, avg_delivery_delay                       (NaN where not applicable -- not imputed)

    Analytical population: the final population is the INTERSECTION of
    customers with usable RFM data and customers with a defined (non-NaN)
    category-share composition. This is an explicit decision made here,
    not a side effect of a merge -- compute_category_shares() reports
    invalid compositions but does not remove them; this function is what
    removes them, and it reports exactly how many at each stage. The
    full stage-by-stage counts are attached to the returned DataFrame as
    `.attrs["population_report"]`, the category selection diagnostics as
    `.attrs["category_selection"]`, and the final category column names
    as `.attrs["category_columns"]` / `.attrs["category_clr_columns"]`.
    """
    orders = valid_orders(tables)
    monetary = order_monetary(tables["order_items"])
    cat_spend = order_category_spend(
        tables["order_items"], tables["products"], tables["category_translation"]
    )
    reviews = deduplicated_order_reviews(tables["order_reviews"])

    stages: dict[str, int] = {}
    stages["delivered_order_customers"] = orders["customer_unique_id"].nunique()

    rfm = compute_rfm(orders, monetary, reference_date=reference_date)
    stages["rfm_eligible_customers"] = len(rfm)

    selected_categories, category_selection_diagnostics = select_categories_for_coverage(
        cat_spend, orders, target_coverage=target_spend_coverage
    )
    categories = compute_category_shares(cat_spend, orders, selected_categories, delta=delta)
    stages["category_table_customers"] = len(categories)

    category_cols = [c for c in categories.columns if c.startswith("category_share__")]
    invalid_composition = int(categories[category_cols].isna().any(axis=1).sum())
    stages["category_invalid_composition"] = invalid_composition

    extras = compute_extras(orders, reviews)

    # The final-population decision: intersect RFM-eligible customers
    # with customers who have a DEFINED category composition. Customers
    # present in `categories` with an all-NaN composition are dropped
    # here, explicitly, rather than inside compute_category_shares().
    categories_valid = categories.dropna(subset=category_cols)

    features = rfm.merge(
        categories_valid, on="customer_unique_id", how="inner", validate="one_to_one"
    ).merge(extras, on="customer_unique_id", how="left", validate="one_to_one")

    stages["final_customers"] = len(features)
    stages["dropped_total"] = stages["rfm_eligible_customers"] - stages["final_customers"]

    if stages["dropped_total"]:
        warnings.warn(
            f"Final customer population is {stages['final_customers']:,}, "
            f"down from {stages['rfm_eligible_customers']:,} RFM-eligible "
            f"customers ({stages['dropped_total']:,} dropped: "
            f"{invalid_composition:,} had an undefined category "
            "composition; any remainder reflects customers present in "
            "RFM but absent from the category table entirely, which "
            "would itself indicate a join problem worth investigating).",
            stacklevel=2,
        )

    features.attrs["population_report"] = stages
    features.attrs["category_selection"] = category_selection_diagnostics
    features.attrs["category_columns"] = category_cols
    features.attrs["category_clr_columns"] = [
        c for c in categories_valid.columns if c.startswith("category_clr__")
    ]
    return features


def validate_features(
    features: pd.DataFrame,
    category_share_cols: list[str] | None = None,
    category_clr_cols: list[str] | None = None,
    tol: float = 1e-6,
) -> None:
    """Validate the mathematical invariants build_customer_features()
    depends on. Raises ValueError on violation -- unlike the population
    exclusions in build_customer_features(), everything checked here is
    expected to hold unconditionally; a failure means something is
    actually broken, not an expected data edge case.

    Mirrors data_loading.py's validate_tables() -- call this
    immediately after build_customer_features() before anything
    downstream depends on the result:

        feats = build_customer_features(tables)
        validate_features(feats, feats.attrs["category_columns"],
                           feats.attrs["category_clr_columns"])

    Checks:
    - customer_unique_id is unique (one row per customer)
    - category shares are non-negative, <= 1, and sum to 1 within `tol`
    - CLR columns sum to 0 within `tol`
    - recency_days >= 0, frequency >= 1, monetary_raw >= 0

    Row-multiplication from joins is NOT re-checked here -- it is
    structurally prevented at join time in this module via
    `merge(..., validate=...)` plus row-count asserts, so by the time a
    DataFrame reaches this function that class of bug cannot be present.
    """
    if category_share_cols is None:
        category_share_cols = [c for c in features.columns if c.startswith("category_share__")]
    if category_clr_cols is None:
        category_clr_cols = [c for c in features.columns if c.startswith("category_clr__")]

    if features["customer_unique_id"].duplicated().any():
        n_dup = int(features["customer_unique_id"].duplicated().sum())
        raise ValueError(f"customer_unique_id is not unique: {n_dup:,} duplicate rows.")

    if category_share_cols:
        shares = features[category_share_cols]
        if (shares < -tol).to_numpy().any() or (shares > 1 + tol).to_numpy().any():
            raise ValueError("category_share__* columns contain values outside [0, 1].")

        row_sums = shares.sum(axis=1)
        bad_sum = (row_sums - 1.0).abs() > tol
        if bad_sum.any():
            raise ValueError(
                f"category_share__* rows do not sum to 1 (tol={tol}): "
                f"{int(bad_sum.sum()):,} violating rows."
            )

    if category_clr_cols:
        clr_sums = features[category_clr_cols].sum(axis=1)
        bad_clr = clr_sums.abs() > tol
        if bad_clr.any():
            raise ValueError(
                f"category_clr__* rows do not sum to 0 (tol={tol}): "
                f"{int(bad_clr.sum()):,} violating rows."
            )

    if (features["recency_days"] < 0).any():
        raise ValueError("recency_days contains negative values.")
    if (features["frequency"] < 1).any():
        raise ValueError("frequency contains values below 1.")
    if (features["monetary_raw"] < 0).any():
        raise ValueError("monetary_raw contains negative values.")


def repeat_customers_only(features: pd.DataFrame) -> pd.DataFrame:
    """Subset the full-base feature table to the repeat-purchase track
    (frequency >= 2), per README's two-track design. Preserves .attrs.
    """
    subset = features[features["frequency"] >= 2].reset_index(drop=True)
    subset.attrs = features.attrs
    return subset


# ---------------------------------------------------------------------------
# PREPROCESSING / LATENT SPACE -- feeds clustering.py
# ---------------------------------------------------------------------------


def fit_latent_space(
    features: pd.DataFrame,
    d_max: int = DEFAULT_D_MAX,
    var_target: float = DEFAULT_VAR_TARGET,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Block-standardize -> MFA-weight -> global PCA.

    Ground truth: theoretical_foundation.md Sec. 2.3-3.3 / 7.1
    (`fit_latent_space` reference implementation). Adapted to this
    project's existing columns: `monetary` is already log1p-transformed
    (not re-applied here) and `category_clr__*` are already CLR-
    transformed (reused directly rather than recomputed from raw shares).

    Blocks: [recency_days, frequency, monetary] (RFM) /
    [category_clr__*] (category) / [avg_review_score, avg_delivery_delay]
    (extras).

    Returns Z (theoretical_foundation.md Sec. 3.3.2) -- the single latent
    space BOTH K-Means and DBSCAN cluster in (Sec. 4, Sec. 5.1.2: "always
    within the same latent space Z"). The intermediate MFA-weighted,
    pre-PCA matrix (A_MFA) is NOT returned -- it is not a space anything
    clusters in, only an internal step toward Z, so there is no reason
    for a caller to hold onto it.

    Runs on COMPLETE CASES ONLY -- MFA/PCA need a dense matrix, and
    avg_review_score / avg_delivery_delay contain NaN by design (see
    module docstring). Every fitted quantity (scaler, sigma_1, PCA) is
    estimated on exactly the rows used here, consistent with
    theoretical_foundation.md Sec. 5.3.1 (re-fit on resample, not a
    frozen transform reused across subsamples).

    Returns:
        Z: (n_complete, d_star) latent space -- the clustering input.
        customer_ids: customer_unique_id for each row of Z, in order.
        diagnostics: n_customers_used/excluded, each block's sigma_1
            (first singular value -- MFA weight is 1/sigma_1), d90,
            d_star, and realized cumulative variance at d_star (per
            theoretical_foundation.md Sec. 3.3.1: "always report
            realized variance", since the d_max cap can keep it below
            var_target -- it does not, at this project's calibrated
            defaults; diagnostics reports `cap_binding` so that stays
            visible rather than assumed).
    """
    category_clr_cols = [c for c in features.columns if c.startswith("category_clr__")]
    complete = features.dropna(subset=["avg_review_score", "avg_delivery_delay"])

    rfm_block = complete[["recency_days", "frequency", "monetary"]].to_numpy()
    cat_block = complete[category_clr_cols].to_numpy()
    extras_block = complete[["avg_review_score", "avg_delivery_delay"]].to_numpy()

    blocks_std = [StandardScaler().fit_transform(b) for b in (rfm_block, cat_block, extras_block)]
    sigma1 = [float(np.linalg.svd(b, compute_uv=False)[0]) for b in blocks_std]
    mfa_weighted = np.hstack([b / s for b, s in zip(blocks_std, sigma1)])

    pca = PCA().fit(mfa_weighted)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cum_var, var_target) + 1)
    d_star = min(d90, d_max)
    Z = pca.transform(mfa_weighted)[:, :d_star]

    diagnostics = {
        "n_customers_used": len(complete),
        "n_customers_excluded_missing_extras": len(features) - len(complete),
        "block_sigma1": {"rfm": sigma1[0], "category": sigma1[1], "extras": sigma1[2]},
        "n_raw_features": mfa_weighted.shape[1],
        "d90": d90,
        "d_max": d_max,
        "d_star": d_star,
        "var_target": var_target,
        "realized_variance": float(cum_var[d_star - 1]),
        "cap_binding": d90 > d_max,
    }
    return Z, complete["customer_unique_id"].to_numpy(), diagnostics


def eps_from_k_distance(Z: np.ndarray, min_samples: int) -> float:
    """k-distance elbow (Kneedle). Ground truth verbatim:
    theoretical_foundation.md Sec. 4.2.2 / 7.1. sklearn's DBSCAN counts
    the point itself, so `min_samples` neighbours are queried including
    self, to match it.
    """
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    distances = np.sort(nn.kneighbors(Z)[0][:, -1])
    knee = KneeLocator(
        np.arange(len(distances)), distances, curve="convex", direction="increasing"
    ).knee
    idx = knee if knee is not None else int(0.95 * len(distances))
    return float(distances[idx])


def preprocessing_diagnostics(features: pd.DataFrame) -> dict:
    """Bundle the diagnostic numbers clustering.py's parameters depend
    on, computed once, here, so clustering.py only fits models.

    Returns a dict with:
        latent_space: fit_latent_space() diagnostics (d_star, realized
            variance, MFA block sigma_1 values, complete-case count).
        dbscan_eps: eps + min_samples, computed ONCE, in the single
            latent space Z (theoretical_foundation.md Sec. 3.3.2/4:
            both K-Means and DBSCAN cluster in the same Z -- there is no
            second, separate DBSCAN space to compute eps for).
        monetary_skew: raw-vs-log1p skew comparison.
        category_selection: already attached to features.attrs, included
            here too so this dict is a self-contained report.
    """
    Z, _, latent_diagnostics = fit_latent_space(features)

    min_samples = MIN_SAMPLES_MULTIPLIER * Z.shape[1]
    eps = eps_from_k_distance(Z, min_samples)

    rfm_cols_present = {"recency_days", "frequency", "monetary_raw", "monetary"}
    skew_report = (
        monetary_skew_report(features)
        if rfm_cols_present.issubset(features.columns)
        else None
    )

    return {
        "latent_space": latent_diagnostics,
        "dbscan_eps": {"eps": eps, "min_samples": min_samples},
        "monetary_skew": skew_report,
        "category_selection": features.attrs.get("category_selection"),
    }
