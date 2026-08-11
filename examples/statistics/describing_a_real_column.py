"""What to compute first when you meet a column.

Centre, spread, shape, and extremes, in one pass. The order matters for interpretation:
the skew tells you whether to report the mean or the median, so compute both before you
decide which one goes in the summary.

    python examples/statistics/describing_a_real_column.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    summary = orders.agg(
        n=bt.count(),
        mean=col("o_totalprice").mean(),
        median=bt.median(col("o_totalprice")),
        std=bt.std(col("o_totalprice")),
        iqr=bt.iqr(col("o_totalprice")),
        skew=bt.skewness(col("o_totalprice")),
        low=col("o_totalprice").min(),
        high=col("o_totalprice").max(),
    ).to_pydict()
    print({name: round(value[0], 2) for name, value in summary.items()})

    assert summary["low"][0] < summary["median"][0] < summary["high"][0]
    assert summary["std"][0] > 0
    assert summary["iqr"][0] > 0

    # Right-skewed, so the mean sits above the median and the median is the honest
    # single number to report.
    assert summary["skew"][0] > 0
    assert summary["mean"][0] > summary["median"][0]

    # The IQR is a more robust spread than the standard deviation for this shape.
    print(f"std {summary['std'][0]:,.0f} vs iqr {summary['iqr'][0]:,.0f}")
    assert summary["iqr"][0] < summary["high"][0] - summary["low"][0]


if __name__ == "__main__":
    main()
