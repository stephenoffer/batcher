"""The SUM-linearity rewrite (`agg_algebra`) preserves results vs DuckDB.

``SUM(base ± c)`` / ``SUM(base * c)`` collapse onto a shared ``SUM(base)`` (+ ``COUNT``),
with each output derived by projection. The sharp edges are NULLs (``COUNT`` must count only
non-null base, and an all-null base must stay NULL, not fold to ``c*0``), a grouped family,
and floats — checked end to end through the full optimizer against DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": pa.array([1, 1, 2, 2, 3], type=pa.int64()),
            "w": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "n": pa.array([5, None, 7, None, 9], type=pa.int64()),
            "f": pa.array([1.5, 2.5, 3.5, 4.5, 5.5], type=pa.float64()),
        }
    )
    duck.register("t", tbl)
    return bt.from_arrow(tbl)


def test_global_shifted_sum_family(t, duck):
    ds = t.agg(**{f"s{i}": (col("w") + i).sum() for i in range(6)})
    cols = ", ".join(f"SUM(w + {i}) AS s{i}" for i in range(6))
    assert_same(ds.collect(), duck.sql(f"SELECT {cols} FROM t"))


def test_grouped_shifted_sum_family(t, duck):
    ds = t.group_by("g").agg(**{f"s{i}": (col("w") + i).sum() for i in range(6)})
    cols = ", ".join(f"SUM(w + {i}) AS s{i}" for i in range(6))
    assert_same(ds.collect(), duck.sql(f"SELECT g, {cols} FROM t GROUP BY g"))


def test_sub_and_mul_and_mixed(t, duck):
    ds = t.agg(
        a=(col("w") + 3).sum(),
        b=(col("w") - 2).sum(),
        c=(col("w") * 4).sum(),
        d=(7 - col("w")).sum(),
        e=(3 + col("w")).sum(),
    )
    assert_same(
        ds.collect(),
        duck.sql(
            "SELECT SUM(w + 3) AS a, SUM(w - 2) AS b, SUM(w * 4) AS c, "
            "SUM(7 - w) AS d, SUM(3 + w) AS e FROM t"
        ),
    )


def test_null_bearing_base_counts_only_non_null(t, duck):
    # COUNT must count only the non-null rows of `n`; a shift over NULL stays NULL.
    ds = t.group_by("g").agg(**{f"s{i}": (col("n") + i * 10).sum() for i in range(4)})
    cols = ", ".join(f"SUM(n + {i * 10}) AS s{i}" for i in range(4))
    assert_same(ds.collect(), duck.sql(f"SELECT g, {cols} FROM t GROUP BY g"))


def test_float_shifted_sum_family(t, duck):
    ds = t.agg(**{f"s{i}": (col("f") + i).sum() for i in range(5)})
    cols = ", ".join(f"SUM(f + {i}) AS s{i}" for i in range(5))
    assert_same(ds.collect(), duck.sql(f"SELECT {cols} FROM t"))


def test_all_null_base_stays_null(duck):
    tbl = pa.table(
        {"g": pa.array([1, 2], type=pa.int64()), "n": pa.array([None, None], type=pa.int64())}
    )
    duck.register("z", tbl)
    ds = bt.from_arrow(tbl).agg(
        a=(col("n") + 1).sum(), b=(col("n") + 2).sum(), c=(col("n") + 3).sum()
    )
    assert_same(
        ds.collect(), duck.sql("SELECT SUM(n + 1) AS a, SUM(n + 2) AS b, SUM(n + 3) AS c FROM z")
    )
