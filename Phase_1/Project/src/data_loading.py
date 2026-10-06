"""Olist dataset acquisition, loading, and structural validation.

This module is responsible only for:
- resolving/downloading the Olist dataset,
- loading the Olist CSV files into named DataFrames,
- validating required columns and basic relational assumptions.

Customer identity semantics: `customer_unique_id` is the persistent
customer identifier across multiple orders and is therefore the key
to use for customer-level aggregation in feature engineering.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
import warnings


# ---------------------------------------------------------------------------
# DATASET / TABLE CONSTANTS
# ---------------------------------------------------------------------------

DATASET_HANDLE = "olistbr/brazilian-ecommerce"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Logical table name -> exact Olist CSV filename.
TABLE_FILES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# Required columns for each table.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "customers": {
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    },
    "orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "order_items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    },
    "order_payments": {
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    },
    "order_reviews": {
        "review_id",
        "order_id",
        "review_score",
        "review_creation_date",
        "review_answer_timestamp",
    },
    "products": {
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    },
    "sellers": {
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    },
    "geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    },
    "category_translation": {
        "product_category_name",
        "product_category_name_english",
    },
}


# Timestamp columns are parsed during loading so downstream code receives
# actual datetime values instead of strings.
DATE_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


# Columns that must be populated for a table to have a valid key.
NON_NULL_KEY_COLUMNS: dict[str, set[str]] = {
    "customers": {
        "customer_id",
        "customer_unique_id",
    },
    "orders": {
        "order_id",
        "customer_id",
    },
    "order_items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
    },
    "order_payments": {
        "order_id",
        "payment_sequential",
    },
    "order_reviews": {
        "review_id",
        "order_id",
    },
    "products": {
        "product_id",
    },
    "sellers": {
        "seller_id",
    },
    "geolocation": {
        "geolocation_zip_code_prefix",
    },
    "category_translation": {
        "product_category_name",
        "product_category_name_english",
    },
}


# Keys that should be unique at the expected table grain.
#
# Tables such as order_payments and order_items contain multiple rows
# per order, so their composite keys are used instead of order_id alone.
UNIQUE_KEYS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "order_reviews": ["review_id", "order_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name"],
}


# ---------------------------------------------------------------------------
# DATASET ACQUISITION / PATH RESOLUTION
# ---------------------------------------------------------------------------


def _find_table_directory(data_dir: Path) -> Path | None:
    """Return a directory containing all required Olist CSV files.

    Searches the supplied directory first, then nested directories. This
    makes the loader tolerant of the exact directory structure returned
    by KaggleHub without hard-coding its cache/output layout.
    """
    required_files = set(TABLE_FILES.values())

    if data_dir.is_file():
        data_dir = data_dir.parent

    if not data_dir.exists():
        return None

    # Prefer the supplied directory itself.
    if required_files.issubset({path.name for path in data_dir.glob("*.csv")}):
        return data_dir

    # Fall back to nested directories.
    for candidate in data_dir.rglob("*"):
        if not candidate.is_dir():
            continue

        csv_names = {path.name for path in candidate.glob("*.csv")}

        if required_files.issubset(csv_names):
            return candidate

    return None


def download_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download the Olist dataset once and resolve its local directory.

    If all required CSVs already exist under `data_dir`, no download is
    performed.

    Args:
        data_dir: Directory in which the raw Olist files should live.

    Returns:
        Path to the directory containing all required Olist CSV files.

    Raises:
        FileNotFoundError:
            If the dataset download completes but the expected files
            cannot be found.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Reuse an already downloaded dataset.
    existing_dir = _find_table_directory(data_dir)

    if existing_dir is not None:
        return existing_dir

    # Download into the project-managed raw-data directory.
    downloaded_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(data_dir),
        )
    )

    resolved_dir = _find_table_directory(downloaded_path)

    if resolved_dir is None:
        # KaggleHub may return a path different from the target directory,
        # so check the target directory as a final resolution step.
        resolved_dir = _find_table_directory(data_dir)

    if resolved_dir is None:
        expected = ", ".join(sorted(TABLE_FILES.values()))

        raise FileNotFoundError(
            "Olist dataset download completed, but the required CSV files "
            f"could not be resolved. Expected: {expected}"
        )

    return resolved_dir


# ---------------------------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------------------------


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the Olist CSV files into named DataFrames.

    Args:
        data_dir: Directory containing the Olist CSV files.

    Returns:
        Mapping of logical table names to DataFrames.

    Raises:
        FileNotFoundError:
            If a required table is missing.
    """
    data_dir = Path(data_dir)

    tables: dict[str, pd.DataFrame] = {}

    for table_name, filename in TABLE_FILES.items():
        file_path = data_dir / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required Olist table not found: {file_path}"
            )

        tables[table_name] = pd.read_csv(
            file_path,
            parse_dates=DATE_COLUMNS.get(table_name),
            low_memory=False,
        )

    return tables


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------


def _validate_required_columns(
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """Validate that a table contains all expected columns."""
    expected = REQUIRED_COLUMNS[table_name]
    actual = set(df.columns)

    missing = expected - actual

    if missing:
        raise ValueError(
            f"Table '{table_name}' is missing required columns: "
            f"{sorted(missing)}"
        )


def _validate_non_null_keys(
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """Validate that required key columns contain no null values."""
    for column in NON_NULL_KEY_COLUMNS[table_name]:
        null_count = int(df[column].isna().sum())

        if null_count:
            raise ValueError(
                f"Table '{table_name}' has {null_count:,} null values "
                f"in required key column '{column}'."
            )


def _validate_unique_keys(
    table_name: str,
    df: pd.DataFrame,
) -> None:
    """Validate uniqueness constraints at the expected table grain."""
    key_columns = UNIQUE_KEYS.get(table_name)

    if not key_columns:
        return

    duplicate_count = int(
        df.duplicated(subset=key_columns, keep=False).sum()
    )

    if duplicate_count:
        key_label = ", ".join(key_columns)

        raise ValueError(
            f"Table '{table_name}' violates uniqueness for key "
            f"({key_label}): {duplicate_count:,} duplicate rows found."
        )


def _validate_relations(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Validate basic foreign-key and lookup relationships."""

    customers = tables["customers"]
    orders = tables["orders"]
    order_items = tables["order_items"]
    order_payments = tables["order_payments"]
    reviews = tables["order_reviews"]
    products = tables["products"]
    sellers = tables["sellers"]

    customer_ids = set(customers["customer_id"])
    order_ids = set(orders["order_id"])
    product_ids = set(products["product_id"])
    seller_ids = set(sellers["seller_id"])

    # orders.customer_id -> customers.customer_id
    missing_order_customers = set(orders["customer_id"]) - customer_ids

    if missing_order_customers:
        raise ValueError(
            "Orders reference customer_id values that are absent from "
            f"the customers table: {len(missing_order_customers):,}."
        )

    # order_items.order_id -> orders.order_id
    missing_item_orders = set(order_items["order_id"]) - order_ids

    if missing_item_orders:
        raise ValueError(
            "Order items reference order_id values that are absent from "
            f"the orders table: {len(missing_item_orders):,}."
        )

    # order_items.product_id -> products.product_id
    missing_item_products = set(order_items["product_id"]) - product_ids

    if missing_item_products:
        raise ValueError(
            "Order items reference product_id values that are absent from "
            f"the products table: {len(missing_item_products):,}."
        )

    # order_items.seller_id -> sellers.seller_id
    missing_item_sellers = set(order_items["seller_id"]) - seller_ids

    if missing_item_sellers:
        raise ValueError(
            "Order items reference seller_id values that are absent from "
            f"the sellers table: {len(missing_item_sellers):,}."
        )

    # order_payments.order_id -> orders.order_id
    missing_payment_orders = set(order_payments["order_id"]) - order_ids

    if missing_payment_orders:
        raise ValueError(
            "Order payments reference order_id values that are absent "
            f"from the orders table: {len(missing_payment_orders):,}."
        )

    # order_reviews.order_id -> orders.order_id
    missing_review_orders = set(reviews["order_id"]) - order_ids

    if missing_review_orders:
        raise ValueError(
            "Reviews reference order_id values that are absent from "
            f"the orders table: {len(missing_review_orders):,}."
        )


def _validate_identity_semantics(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Validate the Olist order-level/customer-level identity model.

    `customer_id` is associated one-to-one with an order/customer record.

    `customer_unique_id` is the persistent customer identifier and may
    legitimately occur multiple times because one customer can place
    multiple orders.
    """
    customers = tables["customers"]
    orders = tables["orders"]

    # One customer_id record per customer row in the customers table.
    if not customers["customer_id"].is_unique:
        raise ValueError(
            "`customers.customer_id` must be unique in the Olist "
            "customers table."
        )

    # One customer_id per order in the orders table.
    if not orders["customer_id"].is_unique:
        raise ValueError(
            "`orders.customer_id` must be unique in the Olist dataset; "
            "customer_id is associated with an individual order."
        )

    # customer_unique_id is intentionally NOT required to be unique.
    #
    # Repeated customer_unique_id values represent multiple orders from
    # the same persistent customer and are required for later customer-
    # level aggregation.


def _report_data_quality(tables: dict[str, pd.DataFrame]) -> None:
    """Report notable raw-data quality characteristics.

    These observations do not modify the data or cause loading to fail.
    They are surfaced so that downstream feature engineering can make
    explicit decisions about how to handle them.
    """
    reviews = tables["order_reviews"]
    products = tables["products"]

    # Reviews linked to multiple orders.
    #
    # `review_id` is not a per-review identifier in Olist: a single review
    # (identical score, comment, and timestamps) can legitimately be linked
    # to more than one order_id. This is an expected schema property, not a
    # data-quality defect -- `(review_id, order_id)` is the real key and is
    # enforced above via UNIQUE_KEYS. Surfaced only as an FYI.
    duplicate_review_ids = int(
        reviews["review_id"].duplicated(keep=False).sum()
    )

    if duplicate_review_ids:
        warnings.warn(
            "order_reviews contains "
            f"{duplicate_review_ids:,} rows where a review_id is linked to "
            "more than one order_id. This is expected (the same review can "
            "apply to multiple orders); no action needed here.",
            stacklevel=2,
        )

    # Orders with multiple distinct reviews.
    #
    # Unlike the above, this DOES need a downstream decision: some orders
    # received more than one distinct review over time (e.g. a revised or
    # follow-up review with a different score). Feature engineering must
    # pick one review per order_id before aggregating to customer_unique_id
    # -- the latest review_answer_timestamp is the recommended rule, since
    # it reflects the customer's final recorded sentiment.
    duplicate_order_reviews = int(
        reviews["order_id"].duplicated(keep=False).sum()
    )

    if duplicate_order_reviews:
        warnings.warn(
            "order_reviews contains "
            f"{duplicate_order_reviews:,} rows where an order_id has "
            "multiple distinct reviews (revised/follow-up reviews). "
            "Feature engineering must pick one review per order before "
            "customer-level aggregation (recommended: latest "
            "review_answer_timestamp).",
            stacklevel=2,
        )

    # Missing product category names.
    missing_product_categories = int(
        products["product_category_name"].isna().sum()
    )

    if missing_product_categories:
        warnings.warn(
            "products contains "
            f"{missing_product_categories:,} rows with missing "
            "product_category_name values.",
            stacklevel=2,
        )

    # Missing timestamps in orders.
    orders = tables["orders"]

    timestamp_columns = DATE_COLUMNS.get("orders", [])

    for column in timestamp_columns:
        missing_count = int(orders[column].isna().sum())

        if missing_count:
            warnings.warn(
                f"orders.{column} contains "
                f"{missing_count:,} missing values.",
                stacklevel=2,
            )


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Validate structure and report raw-data quality observations.

    Structural violations raise exceptions. Data-quality observations are
    reported as warnings without modifying the raw tables.
    """
    expected_tables = set(TABLE_FILES)
    actual_tables = set(tables)

    missing_tables = expected_tables - actual_tables

    if missing_tables:
        raise ValueError(
            f"Missing required Olist tables: {sorted(missing_tables)}"
        )

    for table_name in TABLE_FILES:
        df = tables[table_name]

        if df.empty:
            raise ValueError(f"Table '{table_name}' is empty.")

        _validate_required_columns(table_name, df)
        _validate_non_null_keys(table_name, df)
        _validate_unique_keys(table_name, df)

    _validate_relations(tables)
    _validate_identity_semantics(tables)

    _report_data_quality(tables)


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------


def load_olist(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Download, load, and validate the Olist dataset.

    Args:
        data_dir: Project-local directory used for raw Olist data.

    Returns:
        Validated Olist tables keyed by logical table name.
    """
    resolved_dir = download_dataset(data_dir)
    tables = load_tables(resolved_dir)
    validate_tables(tables)

    return tables
