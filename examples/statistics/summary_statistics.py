"""Summary aggregates beyond mean and stddev.

The averages here answer different questions. Geometric mean is the right average for
growth rates, harmonic mean for rates and speeds, RMS for magnitudes that cancel. Using
the arithmetic mean for all three is the most common quiet mistake in a metrics table.

    python examples/statistics/summary_statistics.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    values = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0], "w": [1.0, 1.0, 1.0, 5.0]})

    stats = values.select(
        mean=bt.mean("x"),
        geometric=bt.geometric_mean("x"),
        harmonic=bt.harmonic_mean("x"),
        rms=bt.rms("x"),
        midrange=bt.midrange("x"),
        # Spread.
        value_range=bt.value_range("x"),
        var_pop=bt.var_pop("x"),
        stddev_pop=bt.stddev_pop("x"),
        sem=bt.sem("x"),
        cv=bt.cv("x"),
        # Weighted, when rows are not equally important.
        weighted=bt.weighted_mean("x", "w"),
        weighted_var=bt.weighted_var("x", "w"),
        weighted_std=bt.weighted_std("x", "w"),
        # Data-health aggregates.
        nulls=bt.null_rate("x"),
        non_nulls=bt.non_null_rate("x"),
        distinct_ratio=bt.nunique_ratio("x"),
    ).to_pydict()

    print(stats)

    assert stats["mean"] == [3.75]
    # 1, 2, 4, 8 doubles every step, so the geometric mean is the middle of the ladder.
    assert abs(stats["geometric"][0] - (1 * 2 * 4 * 8) ** 0.25) < 1e-12
    # For positive values: harmonic <= geometric <= arithmetic <= RMS.
    assert stats["harmonic"][0] <= stats["geometric"][0] <= stats["mean"][0] <= stats["rms"][0]
    assert stats["midrange"] == [4.5]
    assert stats["value_range"] == [7.0]
    # Weighting toward the largest value drags the average up.
    assert stats["weighted"][0] > stats["mean"][0]
    assert stats["nulls"] == [0.0]
    assert stats["non_nulls"] == [1.0]
    assert stats["distinct_ratio"] == [1.0]

    # Nulls are excluded from the aggregates but visible in the rate.
    holey = bt.from_pydict({"x": [1.0, None, 3.0, None]})
    h = holey.select(mean=bt.mean("x"), nulls=bt.null_rate("x")).to_pydict()
    print(h)
    assert h["mean"] == [2.0]
    assert h["nulls"] == [0.5]


if __name__ == "__main__":
    main()
