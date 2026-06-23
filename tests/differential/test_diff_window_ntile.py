"""Differential coverage for percent_rank / cume_dist / ntile vs DuckDB.

These no-input ranking-family window functions bind with ``.over(partition_by=…,
order_by=…)`` and must match SQL ``PERCENT_RANK()`` / ``CUME_DIST()`` /
``NTILE(n) OVER (...)``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import cume_dist, ntile, percent_rank

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b", "b"]),
            "v": pa.array([10, 10, 20, 30, 40, 50], type=pa.int64()),
        }
    )


def test_percent_rank_and_cume_dist(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(
            pr=percent_rank().over(partition_by=["g"], order_by=["v"]),
            cd=cume_dist().over(partition_by=["g"], order_by=["v"]),
        )
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, PERCENT_RANK() OVER (PARTITION BY g ORDER BY v) AS pr, "
            "CUME_DIST() OVER (PARTITION BY g ORDER BY v) AS cd FROM t"
        ),
    )


def test_ntile_even_and_remainder(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(
            q2=ntile(2).over(partition_by=["g"], order_by=["v"]),
            q3=ntile(3).over(partition_by=["g"], order_by=["v"]),
        )
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, NTILE(2) OVER (PARTITION BY g ORDER BY v) AS q2, "
            "NTILE(3) OVER (PARTITION BY g ORDER BY v) AS q3 FROM t"
        ),
    )


def test_ntile_global_more_buckets_than_rows(duck):
    from conftest import assert_same

    small = pa.table({"v": pa.array([1, 2], type=pa.int64())})
    out = bt.from_arrow(small).with_columns(q=ntile(5).over(order_by=["v"])).collect()
    duck.register("s", small)
    assert_same(out, duck.sql("SELECT *, NTILE(5) OVER (ORDER BY v) AS q FROM s"))


def test_percent_rank_global(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).with_columns(pr=percent_rank().over(order_by=["v"])).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT *, PERCENT_RANK() OVER (ORDER BY v) AS pr FROM t"))


def test_ntile_requires_positive_n():
    import pytest as _pytest

    from batcher._internal.errors import PlanError

    with _pytest.raises(PlanError):
        ntile(0)
