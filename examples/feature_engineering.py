"""Feature engineering: derive model-ready columns from raw tabular data.

A compact ML-prep pipeline over a small in-memory table. It walks the moves a
feature pipeline makes — derive columns from expressions, scale a numeric column
against a group/global aggregate, bucket a continuous value into tiers, one-hot
encode a category, and impute missing values — all as lazy ``Expr`` work that runs
in the engine, never row by row in Python.

Broadcast aggregates (``col(...).mean().over()``) are computed as their own columns
first, then later steps reference those columns; window/aggregate expressions do not
nest inside scalar expressions.

    python examples/feature_engineering.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col, lit, when


def main() -> None:
    users = bt.from_pydict(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "category": ["a", "b", "a", "c", "b", "a"],
            "age": [25, 40, None, 33, 52, 19],  # a null to impute
            "amount": [100.0, 250.0, 80.0, 300.0, 500.0, 60.0],
        }
    )

    # Broadcast the aggregates each feature needs as plain columns: the global mean
    # age (for imputation), the global min/max amount (for min-max scaling), and the
    # per-category mean amount (a group-relative feature).
    stats = users.with_columns(
        age_mean=col("age").mean().over(),
        amount_min=col("amount").min().over(),
        amount_max=col("amount").max().over(),
        category_amount_mean=col("amount").mean().over(partition_by=["category"]),
    )

    features = stats.with_columns(
        # Imputation: fill the missing age with the column mean.
        age_filled=col("age").fill_null(col("age_mean")),
        # Min-max scaling of amount into [0, 1].
        amount_scaled=(col("amount") - col("amount_min")) / (col("amount_max") - col("amount_min")),
        # Group-relative feature: how far this row sits from its category average.
        amount_vs_category=col("amount") - col("category_amount_mean"),
        # Conditional bucketing into ordinal tiers.
        spend_tier=when(col("amount") >= 300)
        .then(lit("high"))
        .when(col("amount") >= 100)
        .then(lit("mid"))
        .otherwise(lit("low")),
        # One-hot encoding of the category via boolean expressions cast to 0/1.
        category_a=(col("category") == "a").cast("int64"),
        category_b=(col("category") == "b").cast("int64"),
        category_c=(col("category") == "c").cast("int64"),
    ).select(
        "user_id",
        "age_filled",
        "amount_scaled",
        "amount_vs_category",
        "spend_tier",
        "category_a",
        "category_b",
        "category_c",
    )

    result = features.sort("user_id").to_pydict()
    print(result)

    # Imputation filled the missing age with the mean of the other five ages.
    assert result["age_filled"][2] == 33.8
    assert result["age_filled"][0] == 25.0

    # Min-max scaling maps the smallest amount (60) to 0 and the largest (500) to 1.
    assert result["amount_scaled"][5] == 0.0  # user 6, amount 60
    assert result["amount_scaled"][4] == 1.0  # user 5, amount 500

    # Bucketing: 100 → mid, 300 → high, 80 → low.
    assert result["spend_tier"] == ["mid", "mid", "low", "high", "high", "low"]

    # One-hot columns are mutually exclusive and sum to 1 per row.
    onehot = list(
        zip(result["category_a"], result["category_b"], result["category_c"], strict=True)
    )
    assert all(sum(row) == 1 for row in onehot)
    assert onehot[0] == (1, 0, 0)  # category "a"
    assert onehot[3] == (0, 0, 1)  # category "c"


if __name__ == "__main__":
    main()
