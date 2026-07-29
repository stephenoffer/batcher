"""Filling missing values, and keeping the fact that they were missing.

Imputing silently destroys information: "no value recorded" often predicts the target
better than whatever you filled in. ``MissingIndicator`` keeps that signal as its own
column, so impute *and* flag rather than choosing.

    python examples/ml/preprocessing_imputation.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    data = bt.from_pydict(
        {
            "region": ["us", "us", "eu", "eu", "us", "eu"],
            "income": [100.0, None, 50.0, 70.0, 140.0, None],
            "tier": ["gold", None, "silver", None, "gold", "silver"],
        }
    )

    # Flag first, so the indicator sees the original nulls.
    flagged = ml.MissingIndicator("income").fit(data).transform(data).to_pydict()
    print(flagged["income_missing"])
    assert flagged["income_missing"] == [False, True, False, False, False, True]

    # Column-wide statistics.
    mean_filled = ml.SimpleImputer("income", strategy="mean").fit(data).transform(data).to_pydict()
    print("mean filled:", mean_filled["income"])
    assert None not in mean_filled["income"]
    # (100 + 50 + 70 + 140) / 4 = 90
    assert abs(mean_filled["income"][1] - 90.0) < 1e-9

    median_filled = ml.SimpleImputer("income", strategy="median").fit(data).transform(data)
    assert None not in median_filled.to_pydict()["income"]

    # A fixed value, for the cases where zero (or "unknown") is the honest fill.
    const = ml.SimpleImputer("tier", strategy="constant", fill_value="unknown")
    filled = const.fit(data).transform(data).to_pydict()
    print("constant filled:", filled["tier"])
    assert filled["tier"][1] == "unknown"

    # Most-frequent, for categoricals where a mode is the sensible fill.
    modal = ml.SimpleImputer("tier", strategy="most_frequent").fit(data).transform(data).to_pydict()
    assert None not in modal["tier"]

    # Group-aware imputation: fill from the row's own group, not the global statistic.
    grouped = ml.GroupImputer("income", by="region").fit(data).transform(data).to_pydict()
    print("group filled:", grouped["income"])
    assert None not in grouped["income"]
    # Row 1 is a `us` row, so it is filled from the us mean (120), not the global 90.
    assert abs(grouped["income"][1] - 120.0) < 1e-9

    # The recommended shape: flag, then impute, as one chain.
    pipeline = ml.Chain(
        ml.MissingIndicator("income"),
        ml.SimpleImputer("income", strategy="median"),
    ).fit(data)
    out = pipeline.transform(data).to_pydict()
    assert "income_missing" in out
    assert None not in out["income"]


if __name__ == "__main__":
    main()
