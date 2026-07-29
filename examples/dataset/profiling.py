"""Profiling a table you have just been handed.

The first thing to do with unfamiliar data is measure it, not query it. These are the
one-liners that answer "what is in here" before you write a single business rule.

    python examples/dataset/profiling.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    ds = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10.0, 20.0, None, 40.0, 1000.0],
            "grade": ["a", "b", "a", "a", None],
        }
    )

    # Descriptive statistics per numeric column.
    described = ds.describe().to_pydict()
    print("describe:", sorted(described))
    assert len(described) > 0

    # Null accounting.
    nulls = ds.null_count().to_pydict()
    print("nulls:", nulls)
    assert ds.n_null("amount") == 1
    assert ds.has_nulls("amount")
    assert not ds.has_nulls("id")
    assert not ds.all_null("amount")

    # Cardinality.
    assert ds.n_unique("grade") == 2
    assert ds.approx_n_unique("id") >= 1

    # Distribution, exact and approximate.
    print("median:", ds.median("amount"), "p90:", ds.quantile("amount", 0.9))
    assert ds.min("amount") == 10.0
    assert ds.max("amount") == 1000.0
    assert ds.mean("amount") == (10 + 20 + 40 + 1000) / 4
    assert ds.approx_median("amount") is not None
    assert ds.approx_percentile("amount", 50) is not None

    # Frequency of a categorical.
    counts = ds.value_counts("grade").to_pydict()
    print("value_counts:", counts)
    assert len(counts["grade"]) >= 2

    # Correlation between two numerics, and the whole matrix.
    pairs = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    assert abs(pairs.corr("x", "y") - 1.0) < 1e-9
    matrix = pairs.corr_matrix().to_pydict()
    print("corr matrix:", sorted(matrix))
    assert len(matrix) > 0
    cov = pairs.cov_matrix().to_pydict()
    assert len(cov) > 0

    # Columns that never vary carry no information -- drop them before modelling.
    flat = bt.from_pydict({"varies": [1, 2, 3], "same": [7, 7, 7]})
    trimmed = flat.drop_constant_columns().to_pydict()
    print("after dropping constants:", sorted(trimmed))
    assert "same" not in trimmed
    assert "varies" in trimmed

    # Emptiness, which is worth checking before a downstream step assumes rows exist.
    assert not ds.is_empty()
    assert bt.from_pydict({"a": []}).is_empty()

    # A compact overview of the whole table.
    ds.glimpse()
    ds.info()


if __name__ == "__main__":
    main()
