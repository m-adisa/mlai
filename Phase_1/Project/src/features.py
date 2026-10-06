"""Per-customer feature engineering for Olist customer segmentation.

This module is responsible only for:
- aggregating order-level data to the `customer_unique_id` grain
  (see theoretical_foundation.md Sec. 1 for why this matters),
- computing RFM, category-preference, and marketplace-extras features,
- applying the two *feature-level* transforms (log1p on monetary, CLR
  on category shares) that theoretical_foundation.md Sec. 2 specifies.

It does NOT do: per-block StandardScaler, MFA block-weighting, or PCA.
Those operate across the whole concatenated feature matrix and belong
to clustering.py, per theoretical_foundation.md Sec. 2.3-3.3.

Order population: only `order_status == 'delivered'` orders are used
throughout (README "Data filtering rules"). All aggregations in this
module join at a safe grain (item -> product -> category, item -> order
-> customer) and never join order_items against order_payments, which
avoids the row-multiplication bug documented in theoretical_foundation.md
Sec. 1.2.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DELIVERED_STATUS = "delivered"
DEFAULT_TOP_N_CATEGORIES = 10
OTHER_CATEGORY_LABEL = "other"

# Multiplicative-replacement delta for CLR zero handling. Default matches
# theoretical_foundation.md Sec. 2.2.3 / Sec. 7.1 (delta = 1 / C^2, where C
# is the number of category columns including "other"). Exposed as a
# parameter everywhere so the delta-sensitivity sweep the doc recommends
# can be run from the notebook without editing this module.
DEFAULT_DELTA = None  # resolved to 1 / C**2 per-call, C = n_category_cols


# ---------------------------------------------------------------------------
# ORDER-LEVEL PREPARATION
# ---------------------------------------------------------------------------


def valid_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per delivered order, keyed by customer_unique_id.

    Joins orders -> customers (order_id's customer_id is 1:1 with both
    tables, per data_loading.py's identity-semantics validation, so this
    join cannot duplicate rows). Filters to `order_status == 'delivered'`
    (README "Data filtering rules").

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
    many-to-one and cannot duplicate order_items rows.
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

    missing_monetary = int(with_monetary["monetary_total"].isna().sum())
    if missing_monetary:
        warnings.warn(
            f"{missing_monetary:,} delivered orders have no matching "
            "order_items rows and were excluded from monetary/frequency "
            "aggregation.",
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


# ---------------------------------------------------------------------------
# CATEGORY PREFERENCE
# ---------------------------------------------------------------------------


def select_top_categories(
    category_spend: pd.DataFrame,
    orders: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N_CATEGORIES,
) -> list[str]:
    """Top-N category_english labels by total spend, restricted to
    delivered orders (README: "top-10 ... by spend").

    OTHER_CATEGORY_LABEL is excluded from the ranking even if it happens
    to have high total spend -- it is a catch-all bucket, not a real
    category to feature individually.
    """
    in_scope = category_spend[category_spend["order_id"].isin(orders["order_id"])]
    ranked = (
        in_scope[in_scope["category_english"] != OTHER_CATEGORY_LABEL]
        .groupby("category_english")["category_spend"]
        .sum()
        .sort_values(ascending=False)
    )
    return ranked.head(top_n).index.tolist()


def multiplicative_replacement(
    shares: np.ndarray, delta: float | None = None
) -> np.ndarray:
    """Replace zero shares with delta, rescaling non-zeros so rows still
    sum to 1. Verbatim per theoretical_foundation.md Sec. 2.2.3 / 7.1
    (ground truth for this transform).
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
    top_categories: list[str],
    delta: float | None = None,
) -> pd.DataFrame:
    """Per-customer category spend-share vector (raw shares + CLR columns).

    Categories outside `top_categories` are folded into
    OTHER_CATEGORY_LABEL, so every customer's share vector has
    len(top_categories) + 1 dimensions and sums to 1 (customers with zero
    category spend entirely -- see the warning below -- are dropped,
    since a share vector is undefined for them).
    """
    with_customer = category_spend.merge(
        orders[["order_id", "customer_unique_id"]],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    bucketed = with_customer.copy()
    bucketed["category_bucket"] = bucketed["category_english"].where(
        bucketed["category_english"].isin(top_categories), OTHER_CATEGORY_LABEL
    )

    pivot = (
        bucketed.groupby(["customer_unique_id", "category_bucket"])["category_spend"]
        .sum()
        .unstack(fill_value=0.0)
    )

    category_cols = top_categories + [OTHER_CATEGORY_LABEL]
    for col in category_cols:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[category_cols]

    row_totals = pivot.sum(axis=1)
    zero_spend_customers = int((row_totals == 0).sum())
    if zero_spend_customers:
        warnings.warn(
            f"{zero_spend_customers:,} customers have zero total category "
            "spend (likely all-zero-price items) and were dropped from "
            "category-share computation.",
            stacklevel=2,
        )
        pivot = pivot[row_totals > 0]
        row_totals = row_totals[row_totals > 0]

    shares = pivot.div(row_totals, axis=0)

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
    (README "Data filtering rules").
    """
    with_reviews = orders.merge(reviews, on="order_id", how="left", validate="one_to_one")

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
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------


def build_customer_features(
    tables: dict[str, pd.DataFrame],
    top_n_categories: int = DEFAULT_TOP_N_CATEGORIES,
    delta: float | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the full per-customer feature table.

    Returns one row per customer_unique_id with both raw (interpretable,
    for Route-A profiling per theoretical_foundation.md Sec. 6.2.2) and
    feature-level-transformed columns:

        customer_unique_id
        recency_days, frequency, monetary_raw, monetary            (RFM; monetary = log1p(monetary_raw))
        category_share__<cat> ... category_share__other            (raw shares, sum to 1)
        category_clr__<cat> ... category_clr__other                (CLR-transformed, model-ready)
        avg_review_score, avg_delivery_delay                       (NaN where not applicable -- not imputed)

    Per-block StandardScaler, MFA, and PCA are NOT applied here --
    they belong to clustering.py (see module docstring).
    """
    orders = valid_orders(tables)
    monetary = order_monetary(tables["order_items"])
    cat_spend = order_category_spend(
        tables["order_items"], tables["products"], tables["category_translation"]
    )
    reviews = deduplicated_order_reviews(tables["order_reviews"])

    rfm = compute_rfm(orders, monetary, reference_date=reference_date)

    top_categories = select_top_categories(cat_spend, orders, top_n=top_n_categories)
    categories = compute_category_shares(cat_spend, orders, top_categories, delta=delta)

    extras = compute_extras(orders, reviews)

    features = (
        rfm.merge(categories, on="customer_unique_id", how="inner", validate="one_to_one")
        .merge(extras, on="customer_unique_id", how="left", validate="one_to_one")
    )

    dropped = len(rfm) - len(features)
    if dropped:
        warnings.warn(
            f"{dropped:,} customers present in RFM were absent from the "
            "category-share table (zero category spend) and were dropped "
            "from the final feature table.",
            stacklevel=2,
        )

    return features


def repeat_customers_only(features: pd.DataFrame) -> pd.DataFrame:
    """Subset the full-base feature table to the repeat-purchase track
    (frequency >= 2), per README's two-track design.
    """
    return features[features["frequency"] >= 2].reset_index(drop=True)
