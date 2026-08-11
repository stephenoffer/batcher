"""Counting rows, and the cheapest way to answer each kind of count question.

`count` has to visit rows; `width` and `columns` do not. Between those extremes sit the
questions that only need part of the data — a distinct count over one column reads one
column, and an approximate one reads it once into a fixed-size sketch.

    python examples/relational/counting_without_scanning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # Metadata only.
    assert lineitem.width == 16
    assert len(lineitem.columns) == 16

    # Row count: reads the data.
    rows = lineitem.count()
    print("rows:", rows)
    assert rows == lineitem.height
    assert lineitem.shape == (rows, 16)

    # "Are there any rows" needs one row, not all of them.
    # `has_rows` and `is_empty` are properties, not calls.
    assert lineitem.has_rows
    assert not lineitem.is_empty()
    assert lineitem.filter(col("l_quantity") > 10_000).is_empty()

    # A distinct count over one column reads one column.
    exact = lineitem.n_unique("l_shipmode")
    approx = lineitem.agg(n=bt.approx_n_unique(col("l_shipmode"))).to_pydict()["n"][0]
    print(f"ship modes: exact {exact}, approx {approx}")
    assert exact == approx  # low cardinality, so the sketch is exact here

    # On a high-cardinality column the sketch is close, not exact — and much cheaper.
    exact_parts = lineitem.n_unique("l_partkey")
    approx_parts = lineitem.agg(n=bt.approx_n_unique(col("l_partkey"))).to_pydict()["n"][0]
    error = abs(approx_parts - exact_parts) / exact_parts
    print(f"parts: exact {exact_parts}, approx {approx_parts}, error {error:.4%}")
    assert error < 0.05

    # Null counts, per column, in one pass.
    nulls = lineitem.null_count().to_pydict()
    assert all(value == 0 for column in nulls.values() for value in column)


if __name__ == "__main__":
    main()
