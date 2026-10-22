"""Per-customer feature engineering for Olist customer segmentation.

This module is responsible for:
- aggregating order-level data to the `customer_unique_id` grain,
- splitting customers into train and test sets (customer-level, stratified
  on the repeat-purchase flag) before anything population-fitted is computed,
- computing RFM, category-preference, and marketplace-extras features,
- applying the feature-level transforms (log1p on monetary, CLR on
  category shares),
- building the MFA-weighted, PCA-reduced latent space Z that the
  clustering algorithms run in -- two variants, without and with
  block_balance (see LatentSpace / fit_latent_space) -- fitted on train and
  applied unchanged to test, and the diagnostic numbers that parameterize
  them (PCA component count, MFA block eigenvalues, block variance shares,
  DBSCAN eps, the category-coverage threshold, and a monetary skew check),
- reporting which customers are excluded at each stage,
  and validating the mathematical invariants those transforms depend on.

Order population: only `order_status == 'delivered'` orders are used throughout.
All order-level aggregations join at a safe grain (item -> product -> category, item -> order -> customer)

Train/test protocol: the split is made on customer_unique_id. Every
population-fitted quantity (category selection, block scalers, MFA sigma_1,
block-balance norms, PCA, d_star, DBSCAN eps) is estimated on train only.
Test rows only ever go through `LatentSpace.transform`. The one shared
constant is `reference_date` (the dataset's last purchase timestamp): it is a
calendar anchor, not a fitted statistic, and sharing it keeps recency on one
scale and non-negative for both sides.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DELIVERED_STATUS = "delivered"
OTHER_CATEGORY_LABEL = "other"
DEFAULT_TARGET_SPEND_COVERAGE = 0.87
DEFAULT_DELTA = None
DEFAULT_D_MAX = 26
DEFAULT_VAR_TARGET = 0.87
MIN_SAMPLES_MULTIPLIER = 2

TEST_SIZE = 0.45
SPLIT_RANDOM_STATE = 42

RFM_COLUMNS = ("recency_days", "frequency", "monetary")
EXTRAS_COLUMNS = ("avg_review_score", "avg_delivery_delay")
BLOCK_NAMES = ("rfm", "category", "extras")


# ---------------------------------------------------------------------------
# ORDER-LEVEL PREPARATION
# ---------------------------------------------------------------------------

def valid_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per delivered order, keyed by customer_unique_id.

    Columns: order_id, customer_unique_id, order_purchase_timestamp,
    order_estimated_delivery_date, order_delivered_customer_date,
    delivery_delay_days.
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

    Missing or untranslated product_category_name values are mapped to
    OTHER_CATEGORY_LABEL rather than dropped, so their spend still counts
    toward a customer's total (and therefore their category shares still
    sum to 1).
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

    Args:
        orders: output of valid_orders().
        monetary: output of order_monetary().
        reference_date: anchor for recency. Defaults to the max
            order_purchase_timestamp across `orders` (README: "relative
            to dataset max date").
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


def category_coverage_of(
    category_spend: pd.DataFrame,
    orders: pd.DataFrame,
    categories: list[str],
) -> dict:
    """Coverage diagnostics for a PROVIDED category list on this
    population (used for the test side, whose categories are fixed from
    train). Same definition as select_categories_for_coverage: share of
    non-OTHER spend that the listed categories account for.
    """
    in_scope = category_spend[category_spend["order_id"].isin(orders["order_id"])]
    ranked = (
        in_scope[in_scope["category_english"] != OTHER_CATEGORY_LABEL]
        .groupby("category_english")["category_spend"]
        .sum()
    )
    total_spend = float(ranked.sum())
    achieved = (
        float(ranked.reindex(categories).fillna(0.0).sum() / total_spend)
        if total_spend > 0
        else float("nan")
    )
    return {
        "target_coverage": None,
        "n_categories": len(categories),
        "achieved_coverage": achieved,
        "n_categories_total": len(ranked),
        "source": "provided (train selection)",
    }


def multiplicative_replacement(
    shares: np.ndarray, delta: float | None = None
) -> np.ndarray:
    """Replace zero shares with delta, rescaling non-zeros so rows still
    sum to 1. 
    Rows that are entirely NaN pass through as NaN.
    """
    shares = np.asarray(shares, dtype=float)
    delta = 1.0 / shares.shape[1] ** 2 if delta is None else delta
    zero = shares == 0
    n_zero = zero.sum(axis=1, keepdims=True)
    return np.where(zero, delta, (1.0 - n_zero * delta) * shares)


def clr(shares: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform.
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
    whose every order has a null delivered_customer_date get no delay.
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
    *,
    customer_ids: np.ndarray | None = None,
    selected_categories: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full per-customer feature table.

    Returns one row per customer_unique_id with both raw and feature-level-transformed columns:

        customer_unique_id
        recency_days, frequency, monetary_raw, monetary            (RFM; monetary = log1p(monetary_raw))
        category_share__<cat> ... category_share__other            (raw shares, sum to 1; categories chosen to cover target_spend_coverage of total spend)
        category_clr__<cat> ... category_clr__other                (CLR-transformed, model-ready)
        avg_review_score, avg_delivery_delay                       (NaN where not applicable -- not imputed)

    Optional keyword-only arguments (both default to None = original
    behavior, i.e. the whole delivered-order population and categories
    selected from it):

        customer_ids: restrict the population to these customer_unique_ids.
        selected_categories: use this fixed category list instead of
            selecting one from the population. build_train_test_features()
            passes the train-selected list when building the test side.
    """
    orders = valid_orders(tables)
    if customer_ids is not None:
        orders = orders[orders["customer_unique_id"].isin(customer_ids)]
    monetary = order_monetary(tables["order_items"])
    cat_spend = order_category_spend(
        tables["order_items"], tables["products"], tables["category_translation"]
    )
    reviews = deduplicated_order_reviews(tables["order_reviews"])

    stages: dict[str, int] = {}
    stages["delivered_order_customers"] = orders["customer_unique_id"].nunique()

    rfm = compute_rfm(orders, monetary, reference_date=reference_date)
    stages["rfm_eligible_customers"] = len(rfm)

    if selected_categories is None:
        selected_categories, category_selection_diagnostics = select_categories_for_coverage(
            cat_spend, orders, target_coverage=target_spend_coverage
        )
    else:
        selected_categories = list(selected_categories)
        category_selection_diagnostics = category_coverage_of(
            cat_spend, orders, selected_categories
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
    features.attrs["selected_categories"] = list(selected_categories)
    features.attrs["category_columns"] = category_cols
    features.attrs["category_clr_columns"] = [
        c for c in categories_valid.columns if c.startswith("category_clr__")
    ]
    return features


# ---------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------


def split_customers(
    tables: dict[str, pd.DataFrame],
    test_size: float = TEST_SIZE,
    random_state: int = SPLIT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split RFM-eligible customers (>= 1 delivered order with items) into
    train and test customer_unique_id arrays.

    The split is by customer, never by order, so a customer's orders are
    never on both sides. It is stratified on the repeat-purchase flag
    (frequency >= 2, with frequency counted exactly as compute_rfm counts
    it) so the repeat track is represented proportionally in both sides.
    """
    orders = valid_orders(tables)
    monetary = order_monetary(tables["order_items"])
    with_items = orders.merge(
        monetary[["order_id"]], on="order_id", how="inner", validate="one_to_one"
    )
    n_orders = with_items.groupby("customer_unique_id").size().sort_index()

    ids = n_orders.index.to_numpy()
    is_repeat = n_orders.to_numpy() >= 2

    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=is_repeat,
    )
    return train_ids, test_ids


def build_train_test_features(
    tables: dict[str, pd.DataFrame],
    test_size: float = TEST_SIZE,
    random_state: int = SPLIT_RANDOM_STATE,
    target_spend_coverage: float = DEFAULT_TARGET_SPEND_COVERAGE,
    delta: float | None = None,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (train_features, test_features) with no test leakage.

    1. Split customers (split_customers).
    2. Build train features; category selection runs on train orders only.
    3. Build test features with the train-selected category list, so both
       tables have identical category columns.

    reference_date defaults to the dataset-wide last purchase timestamp
    (same anchor build_customer_features(tables) uses), shared by both
    sides. Both tables have the same columns and attrs as the output of
    build_customer_features(), plus attrs["split"].
    """
    train_ids, test_ids = split_customers(tables, test_size, random_state)
    if reference_date is None:
        reference_date = valid_orders(tables)["order_purchase_timestamp"].max()

    train = build_customer_features(
        tables, target_spend_coverage, delta, reference_date, customer_ids=train_ids
    )
    test = build_customer_features(
        tables,
        target_spend_coverage,
        delta,
        reference_date,
        customer_ids=test_ids,
        selected_categories=train.attrs["selected_categories"],
    )

    for role, table, ids in (("train", train, train_ids), ("test", test, test_ids)):
        table.attrs["split"] = {
            "role": role,
            "test_size": test_size,
            "random_state": random_state,
            "n_assigned": int(len(ids)),
        }
    return train, test


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

    Apply to the train and the test table separately: the full base is
    split once, and each side's repeat track is derived from its own side.
    """
    subset = features[features["frequency"] >= 2].reset_index(drop=True)
    subset.attrs = features.attrs
    return subset


# ---------------------------------------------------------------------------
# PREPROCESSING / LATENT SPACE -- feeds clustering.py
# ---------------------------------------------------------------------------


def _projection_energy(
    A: np.ndarray, pca: PCA, d_star: int, edges: np.ndarray
) -> tuple[list[float], list[float]]:
    """Per-block sum of squares of the (PCA-centered) design matrix before
    and after projecting onto the first d_star train components."""
    centered = A - pca.mean_
    loadings = pca.components_[:d_star].T
    kept_matrix = (centered @ loadings) @ loadings.T
    before = [float((centered[:, i:j] ** 2).sum()) for i, j in zip(edges[:-1], edges[1:])]
    kept = [float((kept_matrix[:, i:j] ** 2).sum()) for i, j in zip(edges[:-1], edges[1:])]
    return before, kept


def _variance_stats(before: list[float], kept: list[float], n_rows: int) -> dict:
    return {
        "share_before_pca": {n: x / sum(before) for n, x in zip(BLOCK_NAMES, before)},
        "share_in_Z": {n: x / sum(kept) for n, x in zip(BLOCK_NAMES, kept)},
        "fraction_kept_in_Z": {n: k / b for n, k, b in zip(BLOCK_NAMES, kept, before)},
        "realized_variance": sum(kept) / sum(before),
        "reconstruction_mse": (sum(before) - sum(kept)) / n_rows,
    }


class LatentSpace:
    """Block-standardize -> MFA-weight -> (block balance) -> global PCA,
    fitted on one table and applicable to another.

    fit(train) estimates and stores everything population-dependent: the
    per-block scalers, per-block sigma_1 (MFA weight = 1/sigma_1), the
    block-balance norms (block_balance=True only), the PCA, d90 and d_star.
    transform(other) applies those stored quantities without refitting.

    Complete cases only: rows with NaN avg_review_score or
    avg_delivery_delay are excluded from fit and from transform alike
    (MFA/PCA need a dense matrix; see module docstring).

    After fit(): Z_ (train latent space), customer_ids_ (its row order),
    diagnostics_ (identical in content to the diagnostics
    fit_latent_space() has always returned).
    """

    def __init__(
        self,
        block_balance: bool = False,
        d_max: int = DEFAULT_D_MAX,
        var_target: float = DEFAULT_VAR_TARGET,
    ):
        self.block_balance = block_balance
        self.d_max = d_max
        self.var_target = var_target

    # -- internals ---------------------------------------------------------

    def _complete(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.dropna(subset=list(EXTRAS_COLUMNS))

    def _blocks(self, complete: pd.DataFrame) -> list[np.ndarray]:
        missing = [c for cols in self.block_columns_ for c in cols if c not in complete.columns]
        if missing:
            raise ValueError(f"features is missing columns the latent space was fitted on: {missing}")
        return [complete[cols].to_numpy() for cols in self.block_columns_]

    def _assemble(self, blocks: list[np.ndarray]) -> np.ndarray:
        weighted = [
            scaler.transform(b) / s
            for scaler, b, s in zip(self.scalers_, blocks, self.sigma1_)
        ]
        if self.block_balance:
            weighted = [w / n for w, n in zip(weighted, self.balance_norms_)]
        return np.hstack(weighted)

    # -- fit / transform ---------------------------------------------------

    def fit(self, features: pd.DataFrame) -> "LatentSpace":
        category_clr_cols = [c for c in features.columns if c.startswith("category_clr__")]
        self.block_columns_ = [list(RFM_COLUMNS), category_clr_cols, list(EXTRAS_COLUMNS)]

        complete = self._complete(features)
        blocks = self._blocks(complete)

        self.scalers_ = [StandardScaler().fit(b) for b in blocks]
        blocks_std = [s.transform(b) for s, b in zip(self.scalers_, blocks)]
        self.sigma1_ = [float(np.linalg.svd(b, compute_uv=False)[0]) for b in blocks_std]
        weighted = [b / s for b, s in zip(blocks_std, self.sigma1_)]
        self.balance_norms_ = (
            [float(np.sqrt((w ** 2).sum())) for w in weighted] if self.block_balance else None
        )

        A = self._assemble(blocks)
        self.edges_ = np.cumsum([0] + [b.shape[1] for b in blocks])

        self.pca_ = PCA().fit(A)
        cum_var = np.cumsum(self.pca_.explained_variance_ratio_)
        self.d90_ = int(np.searchsorted(cum_var, self.var_target) + 1)
        self.d_star_ = min(self.d90_, self.d_max)

        self.Z_ = self.pca_.transform(A)[:, : self.d_star_]
        self.customer_ids_ = complete["customer_unique_id"].to_numpy()

        before, kept = _projection_energy(A, self.pca_, self.d_star_, self.edges_)
        self.train_stats_ = _variance_stats(before, kept, len(A))
        block_variance = {
            k: self.train_stats_[k]
            for k in ("share_before_pca", "share_in_Z", "fraction_kept_in_Z")
        }

        self.diagnostics_ = {
            "block_balance": self.block_balance,
            "n_customers_used": len(complete),
            "n_customers_excluded_missing_extras": len(features) - len(complete),
            "block_sigma1": dict(zip(BLOCK_NAMES, self.sigma1_)),
            "block_variance": block_variance,
            "n_raw_features": A.shape[1],
            "d90": self.d90_,
            "d_max": self.d_max,
            "d_star": self.d_star_,
            "var_target": self.var_target,
            "realized_variance": float(cum_var[self.d_star_ - 1]),
            "cap_binding": self.d90_ > self.d_max,
        }
        return self

    def transform(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Project another table into the fitted space. Nothing is refitted.

        Returns (Z, customer_ids) for the complete-case rows of `features`.
        """
        complete = self._complete(features)
        A = self._assemble(self._blocks(complete))
        Z = self.pca_.transform(A)[:, : self.d_star_]
        return Z, complete["customer_unique_id"].to_numpy()

    def diagnose(self, features: pd.DataFrame) -> dict:
        """Held-out latent-space diagnostics, train vs. `features`.

        The train PCA is applied to `features`; how much of its variance
        the d_star train components capture, and how well each block is
        represented, is compared with the same numbers on train.
        """
        complete = self._complete(features)
        A = self._assemble(self._blocks(complete))
        before, kept = _projection_energy(A, self.pca_, self.d_star_, self.edges_)
        test = _variance_stats(before, kept, len(A))
        train = self.train_stats_

        def pair(key: str) -> dict:
            return {n: {"train": train[key][n], "test": test[key][n]} for n in BLOCK_NAMES}

        return {
            "n_customers_used": len(complete),
            "n_customers_excluded_missing_extras": len(features) - len(complete),
            "d_star": self.d_star_,
            "realized_variance": {
                "train": train["realized_variance"],
                "test": test["realized_variance"],
                "gap": train["realized_variance"] - test["realized_variance"],
            },
            # Per-row reconstruction error of test relative to train (1.0 =
            # same; > 1 means the train PCA represents test rows worse).
            "reconstruction_error_ratio": (
                test["reconstruction_mse"] / train["reconstruction_mse"]
            ),
            "block_variance": {
                "share_before_pca": pair("share_before_pca"),
                "share_in_Z": pair("share_in_Z"),
                "fraction_kept_in_Z": pair("fraction_kept_in_Z"),
            },
        }


def fit_latent_space(
    features: pd.DataFrame,
    d_max: int = DEFAULT_D_MAX,
    var_target: float = DEFAULT_VAR_TARGET,
    block_balance: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Block-standardize -> MFA-weight -> (block balance) -> global PCA.

    block_balance=False is the math doc's pipeline: MFA equalizes each
    block's first axis only, so the 26-column category block still holds
    most of the total variance.

    block_balance=True adds one step after MFA weighting: every block is
    rescaled to the same total variance (a third each). This goes beyond
    the math doc. Rescaling each block by its own total cancels MFA's
    per-block scaling, so the result is consensus-PCA-style block scaling
    (Westerhuis, Kourti & MacGregor, 1998), not standard MFA.

    Global PCA runs in both cases. Equal total variance is not equal
    influence on the clustering: a block spread over many axes has weak
    variance per axis, and PCA/K-Means follow the strong axes.

    Runs on COMPLETE CASES ONLY -- MFA/PCA need a dense matrix, and
    avg_review_score / avg_delivery_delay contain NaN by design (see
    module docstring). Every fitted quantity (scaler, sigma_1, PCA) is
    estimated on exactly the rows used here.

    This is a thin wrapper over LatentSpace().fit(features); call
    LatentSpace directly when the fitted space must be applied to other
    rows (e.g. the test set).

    Returns:
        Z: (n_complete, d_star) latent space -- the clustering input.
        customer_ids: customer_unique_id for each row of Z, in order.
        diagnostics: n_customers_used/excluded, each block's sigma_1
            (first singular value -- MFA weight is 1/sigma_1), d90,
            d_star, realized cumulative variance at d_star, and
            block_variance: each block's share of the total variance
            before PCA, its share of the variance kept in Z, and the
            fraction of its own variance that survives the PCA cut.
    """
    model = LatentSpace(block_balance=block_balance, d_max=d_max, var_target=var_target)
    model.fit(features)
    return model.Z_, model.customer_ids_, model.diagnostics_


def eps_from_k_distance(Z: np.ndarray, min_samples: int) -> float:
    """k-distance elbow (Kneedle).
    """
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    distances = np.sort(nn.kneighbors(Z)[0][:, -1])
    knee = KneeLocator(
        np.arange(len(distances)), distances, curve="convex", direction="increasing"
    ).knee
    idx = knee if knee is not None else int(0.95 * len(distances))
    return float(distances[idx])


def preprocessing_diagnostics(
    features: pd.DataFrame,
    test_features: pd.DataFrame | None = None,
    *,
    verbose: bool = True,
) -> dict:
    """Bundle the diagnostic numbers that clustering parameters depend on,
    for both latent spaces: "mfa_space" (no block_balance) and
    "balanced_space" (block_balance=True). Each space gets its own latent
    diagnostics and its own DBSCAN (eps, min_samples), since both depend
    on that space's dimensionality. Metrics from the two spaces are not
    comparable with each other.

    `features` is the TRAIN table: every fitted number here comes from it.
    If `test_features` is given, each space also gets a
    "heldout_latent_space" section (train PCA applied to test, train vs.
    test side by side) and a "split" section is added. With
    test_features=None the output is unchanged from before the split.

    Prints a formatted report when ``verbose`` is True (the default).
    Returns the underlying dict either way -- assign the result in a
    notebook (``diagnostics = preprocessing_diagnostics(train_feats, test_feats)``) so the
    dict is not echoed under the printed report.
    """
    diagnostics = {}
    for name, balance in (("mfa_space", False), ("balanced_space", True)):
        model = LatentSpace(block_balance=balance).fit(features)
        min_samples = MIN_SAMPLES_MULTIPLIER * model.Z_.shape[1]
        diagnostics[name] = {
            "latent_space": model.diagnostics_,
            "dbscan_eps": {
                "eps": eps_from_k_distance(model.Z_, min_samples),
                "min_samples": min_samples,
            },
        }
        if test_features is not None:
            diagnostics[name]["heldout_latent_space"] = model.diagnose(test_features)

    rfm_cols_present = {"recency_days", "frequency", "monetary_raw", "monetary"}
    diagnostics["monetary_skew"] = (
        monetary_skew_report(features)
        if rfm_cols_present.issubset(features.columns)
        else None
    )
    diagnostics["category_selection"] = features.attrs.get("category_selection")

    if test_features is not None:
        split_attrs = features.attrs.get("split") or {}
        diagnostics["split"] = {
            "test_size": split_attrs.get("test_size"),
            "random_state": split_attrs.get("random_state"),
            "n_train_customers": len(features),
            "n_test_customers": len(test_features),
            "test_category_coverage": test_features.attrs.get("category_selection"),
        }

    if verbose:
        _render_diagnostics(diagnostics)
    return diagnostics


def _render_diagnostics(diagnostics: dict) -> None:
    """Print a nested diagnostics dict in an aligned, grouped layout."""
    print("Preprocessing diagnostics")
    print("=" * 60)
    for section, values in diagnostics.items():
        _print_section(values, title=f"[{section}]")
        print()


def _print_section(d, indent: int = 0, title: str | None = None) -> None:
    pad = "  " * indent
    if title is not None:
        print(f"{pad}{title}")

    if not isinstance(d, dict):
        print(f"{pad}{d}")
        return

    key_w = max((len(str(k)) for k in d), default=0)
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_section(value, indent + 1)
        else:
            print(f"{pad}{key:<{key_w}}  {_fmt_scalar(value)}")


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)
