"""Is this column symmetric, skewed, or heavy-tailed?

Shape decides which summary is honest. On a skewed column the mean is not the typical
value, and a normality-assuming test is not valid. These aggregates answer that question
before you pick the summary rather than after someone questions the dashboard.

    python examples/statistics/distribution_shape.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    symmetric = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
    right_skewed = bt.from_pydict({"x": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 5.0, 9.0, 40.0]})

    def shape(ds: bt.Dataset) -> dict[str, list[float]]:
        return ds.select(
            skew=bt.skewness("x"),
            bowley=bt.bowley_skew("x"),
            pearson_mode=bt.pearson_mode_skew("x"),
            kurtosis=bt.kurtosis("x"),
            moors=bt.moors_kurtosis("x"),
            jarque_bera=bt.jarque_bera("x"),
            # Moment-derived spread summaries.
            snr=bt.signal_to_noise("x"),
            dispersion_index=bt.index_of_dispersion("x"),
            relative_range=bt.relative_range("x"),
            studentized_range=bt.studentized_range("x"),
        ).to_pydict()

    s = shape(symmetric)
    r = shape(right_skewed)
    print("symmetric   :", s)
    print("right-skewed:", r)

    # A symmetric column has ~zero skew; the skewed one is clearly positive.
    assert abs(s["skew"][0]) < 1e-9
    assert abs(s["bowley"][0]) < 1e-9
    assert r["skew"][0] > 1.0
    assert r["bowley"][0] > 0.0
    # Jarque-Bera rises with departure from normality.
    assert r["jarque_bera"][0] > s["jarque_bera"][0]
    # A long right tail inflates the variance relative to the mean.
    assert r["dispersion_index"][0] > s["dispersion_index"][0]

    # The decision this drives: report the median, not the mean, on the skewed column.
    compare = right_skewed.select(mean=bt.mean("x"), median=bt.median("x")).to_pydict()
    print(compare)
    assert compare["mean"][0] > compare["median"][0]


if __name__ == "__main__":
    main()
