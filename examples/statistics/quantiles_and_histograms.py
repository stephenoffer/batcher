"""Quantiles, histograms, and the exact-versus-approximate trade.

Exact quantiles need the whole column ordered. Sketch-backed ones need bounded memory and
answer within a known error, which is what makes them usable on a column that does not fit
in memory. Know which one you are getting.

    python examples/statistics/quantiles_and_histograms.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    latency = bt.from_pydict({"ms": [float(v) for v in range(1, 1001)]})

    exact = latency.select(
        p50=col("ms").quantile(0.5),
        p90=col("ms").quantile(0.9),
        p99=col("ms").quantile(0.99),
        lo=col("ms").min(),
        hi=col("ms").max(),
    ).to_pydict()
    print("exact:", exact)

    assert exact["lo"] == [1.0] and exact["hi"] == [1000.0]
    assert 495 <= exact["p50"][0] <= 505
    assert 895 <= exact["p90"][0] <= 905
    assert 985 <= exact["p99"][0] <= 995

    # Quantiles are monotonic in the probability.
    assert exact["p50"][0] < exact["p90"][0] < exact["p99"][0]

    # Sketch-backed, for a column that does not fit in memory.
    approx = latency.select(
        p50=col("ms").approx_quantile(0.5),
        p90=col("ms").approx_quantile(0.9),
        median=col("ms").approx_median(),
    ).to_pydict()
    print("approx:", approx)
    # Close to exact, not equal to it -- that is the trade.
    assert abs(approx["p50"][0] - exact["p50"][0]) < 50
    assert approx["p50"][0] < approx["p90"][0]

    # Dataset-level shorthands for the same thing.
    assert abs(latency.quantile("ms", 0.5) - exact["p50"][0]) < 1e-9
    assert latency.approx_percentile("ms", 90) is not None

    # Per-group quantiles, in one pass -- the reason these are aggregates.
    mixed = bt.from_pydict(
        {
            "route": ["a"] * 500 + ["b"] * 500,
            "ms": [float(v) for v in range(1, 501)] + [float(v) for v in range(501, 1001)],
        }
    )
    by_route = (
        mixed.group_by("route")
        .agg(p50=col("ms").quantile(0.5), p95=col("ms").quantile(0.95), n=bt.count())
        .sort("route")
        .to_pydict()
    )
    print(by_route)
    assert by_route["route"] == ["a", "b"]
    assert by_route["n"] == [500, 500]
    # Route b is uniformly slower.
    assert by_route["p50"][1] > by_route["p50"][0]

    # A histogram of the distribution, when a few quantiles are not enough.
    buckets = (
        latency.group_by(bucket=(col("ms") / 100).cast("int64"))
        .agg(n=bt.count())
        .sort("bucket")
        .to_pydict()
    )
    print("histogram:", buckets["n"])
    assert sum(buckets["n"]) == 1000


if __name__ == "__main__":
    main()
