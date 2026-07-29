"""Robust spread: quantile-based measures that one outlier cannot move.

Standard deviation is a poor summary of a long-tailed distribution, which describes most
latency and revenue data. These are the quantile-based alternatives: a single extreme row
shifts them barely at all, so a dashboard built on them stops flapping.

    python examples/statistics/robust_dispersion.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    # A tidy distribution, and the same data with one absurd outlier appended.
    clean = bt.from_pydict({"ms": [float(v) for v in range(1, 21)]})
    spiked = bt.from_pydict({"ms": [float(v) for v in range(1, 21)] + [100_000.0]})

    def summarize(ds: bt.Dataset) -> dict[str, list[float]]:
        return ds.select(
            stddev=bt.stddev_pop("ms"),
            median=bt.median("ms"),
            trimean=bt.trimean("ms"),
            midhinge=bt.midhinge("ms"),
            interdecile=bt.interdecile_range("ms"),
            quartile_dispersion=bt.quartile_dispersion("ms"),
            robust_cv=bt.robust_cv("ms"),
            decile_ratio=bt.decile_ratio("ms"),
        ).to_pydict()

    a = summarize(clean)
    b = summarize(spiked)
    print("clean :", a)
    print("spiked:", b)

    # The classical measure moves by orders of magnitude; the robust ones barely budge.
    assert b["stddev"][0] > a["stddev"][0] * 100
    assert b["median"][0] == a["median"][0] + 0.5
    assert abs(b["trimean"][0] - a["trimean"][0]) < 2.0
    assert abs(b["midhinge"][0] - a["midhinge"][0]) < 2.0

    # The robust summaries are all positive spreads on a positive distribution.
    for name in ("interdecile", "quartile_dispersion", "robust_cv", "decile_ratio"):
        assert a[name][0] > 0.0, name


if __name__ == "__main__":
    main()
