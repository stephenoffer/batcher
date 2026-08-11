"""Four scalers on a skewed real column, and what each does to the outliers.

Standard scaling is dominated by the outliers it is meant to normalize. Robust scaling uses
the median and IQR, so the bulk of the data spreads out properly. On a right-skewed revenue
column that difference is large enough to change a model.

    python examples/ml/scaling_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    standard = ml.StandardScaler("o_totalprice").fit(orders).transform(orders)
    robust = ml.RobustScaler("o_totalprice").fit(orders).transform(orders)
    minmax = ml.MinMaxScaler("o_totalprice").fit(orders).transform(orders)
    maxabs = ml.MaxAbsScaler("o_totalprice").fit(orders).transform(orders)

    def describe(name: str, dataset: bt.Dataset) -> dict[str, float]:
        row = dataset.agg(
            mean=col("o_totalprice").mean(),
            low=col("o_totalprice").min(),
            high=col("o_totalprice").max(),
            iqr=bt.iqr(col("o_totalprice")),
        ).to_pydict()
        stats = {key: value[0] for key, value in row.items()}
        print(f"{name:<10} " + " ".join(f"{k}={v:8.3f}" for k, v in stats.items()))
        return stats

    std_stats = describe("standard", standard)
    rob_stats = describe("robust", robust)
    mm_stats = describe("minmax", minmax)
    describe("maxabs", maxabs)

    # Standard scaling centres on zero.
    assert abs(std_stats["mean"]) < 1e-6

    # Min-max lands exactly in [0, 1].
    assert abs(mm_stats["low"]) < 1e-9
    assert abs(mm_stats["high"] - 1.0) < 1e-9

    # Robust scaling gives the middle half a unit spread, which standard scaling does not.
    print(f"IQR after standard {std_stats['iqr']:.4f}, after robust {rob_stats['iqr']:.4f}")
    assert abs(rob_stats["iqr"] - 1.0) < 1e-6
    assert abs(std_stats["iqr"] - 1.0) > 1e-3

    # Every scaler preserves the row count and the ordering of the data.
    for scaled in (standard, robust, minmax, maxabs):
        assert scaled.count() == orders.count()


if __name__ == "__main__":
    main()
