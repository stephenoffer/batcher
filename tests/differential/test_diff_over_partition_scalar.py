"""`over(partition_by=...)` must accept a scalar key without corrupting or hanging.

Both `AggExpr.over` and `WindowExpr.over` normalized their `partition_by`/`order_by`
arguments with a bare ``list(...)``. That silently broke the two most natural scalar
spellings:

* ``over(partition_by="grp")`` → ``list("grp")`` == ``['g', 'r', 'p']``, so the window
  partitioned by three phantom single-character columns (raising "unknown column", or —
  worse — partitioning by the wrong columns if any single-char column happened to exist).
* ``over(partition_by=col("grp"))`` → ``list(expr)`` iterated an `Expr`, which has an
  unbounded ``__getitem__`` and no ``__iter__``, so it looped forever allocating nodes
  (an out-of-memory hang from a completely ordinary call).

`normalize_key_list` wraps a lone ``str``/``Expr`` in a one-element list; `Expr.__iter__`
now raises ``TypeError`` so any other accidental ``list(expr)`` fails fast instead of
exhausting memory. Each spelling must match the explicit-list spelling and DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table({"grp": [1, 1, 2, 2, 2], "v": [10, 20, 30, 40, 50]})
    duck.register("t", tbl)
    return tbl


def test_over_string_partition_key(duck, t):
    """``over(partition_by="grp")`` partitions by column ``grp``, not by ``['g','r','p']``."""
    out = bt.from_arrow(t).with_columns(s=col("v").sum().over(partition_by="grp")).collect()
    assert_same(out, duck.sql("SELECT *, SUM(v) OVER (PARTITION BY grp) AS s FROM t"))


def test_over_expr_partition_key(duck, t):
    """``over(partition_by=col("grp"))`` completes (no OOM hang) and partitions correctly."""
    out = bt.from_arrow(t).with_columns(s=col("v").sum().over(partition_by=col("grp"))).collect()
    assert_same(out, duck.sql("SELECT *, SUM(v) OVER (PARTITION BY grp) AS s FROM t"))


def test_over_scalar_equals_list_spelling(t):
    """The scalar, `Expr`, and explicit-list spellings all produce the same result."""
    ds = bt.from_arrow(t)
    as_str = ds.with_columns(s=col("v").sum().over(partition_by="grp")).collect().to_pydict()
    as_expr = ds.with_columns(s=col("v").sum().over(partition_by=col("grp"))).collect().to_pydict()
    as_list = ds.with_columns(s=col("v").sum().over(partition_by=["grp"])).collect().to_pydict()
    assert as_str == as_list == as_expr


def test_expression_is_not_iterable():
    """A stray ``list(expr)`` raises ``TypeError`` immediately, never loops/OOMs."""
    with pytest.raises(TypeError, match="not iterable"):
        list(col("x"))
