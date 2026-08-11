"""A sweep over the Dataset API: every accessor and metadata method, checked.

The purpose is coverage rather than instruction. If a method is renamed or removed, this
fails — which is the point of running the examples as a release check.

    python examples/dataset/reading_the_whole_surface.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = (
        tpch("orders")
        .select("o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate")
        .head(5_000)
    )

    # Metadata, none of which executes.
    assert orders.width == 5
    assert len(orders.columns) == 5
    assert len(orders.dtypes) == 5
    assert orders.schema.names == orders.columns
    assert list(orders.collect_schema()) == orders.columns

    # Shape, which does.
    assert orders.count() == 5_000
    assert orders.height == 5_000
    assert orders.shape == (5_000, 5)
    assert orders.has_rows
    assert not orders.is_empty()

    # Per-column reductions as frame methods.
    print("min/max/mean:", orders.min("o_totalprice"), orders.max("o_totalprice"))
    assert orders.min("o_totalprice") <= orders.mean("o_totalprice") <= orders.max("o_totalprice")
    assert orders.sum("o_totalprice") > 0
    assert orders.std("o_totalprice") > 0
    assert orders.var("o_totalprice") > 0
    assert orders.median("o_totalprice") > 0
    assert orders.n_unique("o_orderkey") == 5_000
    # `nunique` is the whole-frame form: one count per column, no argument.
    assert set(orders.nunique().to_pydict()) == set(orders.columns)

    # Null and emptiness helpers.
    assert orders.null_count().count() >= 1
    assert not orders.has_nulls("o_orderkey")
    assert orders.drop_nulls().count() == orders.count()
    # `drop_empty` names the column whose empties to drop.
    assert orders.drop_empty("o_orderstatus").count() <= orders.count()

    # Row access.
    assert len(orders.first()) == 5
    assert len(orders.last()) == 5
    assert orders.head(3).count() == 3
    assert orders.tail(3).count() == 3
    assert orders.slice(10, 5).count() == 5
    assert orders.sample(n=10, seed=1).count() == 10
    assert orders.limit(7).count() == 7
    assert orders.gather_every(100).count() == 50

    # Reshaping.
    assert orders.reverse().count() == orders.count()
    assert orders.rename({"o_orderkey": "id"}).columns[0] == "id"
    assert orders.drop("o_custkey").width == 4
    assert orders.with_row_index().width == 6
    assert orders.select_dtypes("int64").width == 2

    # Conversions.
    assert orders.to_arrow().num_rows == 5_000
    assert len(orders.to_pylist()) == 5_000
    assert set(orders.to_pydict()) == set(orders.columns)
    assert len(orders.to_pandas()) == 5_000
    assert set(orders.to_numpy()) == set(orders.columns)

    # Summaries.
    assert orders.describe().count() > 0
    assert orders.value_counts("o_orderstatus").count() <= 3
    orders.glimpse()

    # Plan surface.
    assert isinstance(orders.explain(), str)
    assert orders.lazy().count() == orders.count()
    assert orders.copy().count() == orders.count()
    assert orders.pipe(lambda d: d.filter(col("o_totalprice") > 0)).count() == orders.count()
    assert orders.equals(orders.copy())

    print("Dataset surface sweep passed")
    assert bt is not None


if __name__ == "__main__":
    main()
