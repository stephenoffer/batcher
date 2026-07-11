"""Filtered-count metadata shortcuts equal DuckDB — over a real Parquet footer and memory.

`answer_filter_count` / `answer_filter_is_empty` / `answer_filter_any` answer a
`count()`/`is_empty()`/`any()` over a `Filter` from EXACT footer statistics when provable
(a null predicate, a provably-empty comparison, or a tautology) and return `None`
otherwise. Either way the count MUST equal DuckDB's executed answer. Each case is checked
over a Parquet file (real footer drives the shortcut) and in memory (the shortcut mostly
falls back), and the public executed `count()` is always cross-checked against DuckDB so
both the metadata path and the fallback are proven correct. Covers NULLs, an all-null
column, a constant column, empty results, and int/float/str/date/bool edges.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import core
from batcher.api.orchestration import collect_source_stats
from batcher.kyber.metadata_filter_count import (
    answer_filter_any,
    answer_filter_count,
    answer_filter_is_empty,
)

pytestmark = pytest.mark.differential

_TABLE = pa.table(
    {
        "i": pa.array([3, 1, 2, None, 5], type=pa.int64()),
        "f": pa.array([1.5, None, 2.5, 2.5, 4.0], type=pa.float64()),
        "s": pa.array(["b", "a", None, "a", "c"], type=pa.string()),
        "d": pa.array(
            [
                datetime.date(2024, 1, 3),
                datetime.date(2024, 1, 1),
                None,
                None,
                datetime.date(2024, 1, 5),
            ],
            type=pa.date32(),
        ),
        "b": pa.array([True, False, True, None, True], type=pa.bool_()),
        "allnull": pa.array([None] * 5, type=pa.int64()),
        "k": pa.array([7, 7, 7, 7, 7], type=pa.int64()),
    }
)


@pytest.fixture
def pq_path(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(_TABLE, path)
    return path


def _duck(con):
    con.register("t", _TABLE)
    return con


def _stats(ds):
    return collect_source_stats(ds._sources, core.default_hub())


def _count(ds):
    return answer_filter_count(ds._plan, ds._sources, _stats(ds), core.default_hub())


# (name, batcher predicate, DuckDB WHERE, must-fire-from-footer)
_CASES = [
    ("i_is_null", lambda: bt.col("i").is_null(), "i IS NULL", True),
    ("i_is_not_null", lambda: bt.col("i").is_not_null(), "i IS NOT NULL", True),
    ("i_not_is_null", lambda: ~bt.col("i").is_null(), "NOT (i IS NULL)", True),
    ("i_not_is_not_null", lambda: ~bt.col("i").is_not_null(), "NOT (i IS NOT NULL)", True),
    ("allnull_is_null", lambda: bt.col("allnull").is_null(), "allnull IS NULL", True),
    ("allnull_is_not_null", lambda: bt.col("allnull").is_not_null(), "allnull IS NOT NULL", True),
    ("d_is_null", lambda: bt.col("d").is_null(), "d IS NULL", True),
    ("b_is_null", lambda: bt.col("b").is_null(), "b IS NULL", True),
    # Provably-empty comparisons (literal entirely outside the EXACT [min, max]).
    ("i_gt_max", lambda: bt.col("i") > 100, "i > 100", True),
    ("i_ge_over", lambda: bt.col("i") >= 6, "i >= 6", True),
    ("i_lt_min", lambda: bt.col("i") < 0, "i < 0", True),
    ("i_le_under", lambda: bt.col("i") <= 0, "i <= 0", True),
    ("i_eq_absent", lambda: bt.col("i") == 999, "i = 999", True),
    ("i_rev_gt", lambda: bt.lit(100) < bt.col("i"), "100 < i", True),
    ("f_gt_max", lambda: bt.col("f") > 9.0, "f > 9.0", True),
    ("k_ne_const", lambda: bt.col("k") != 7, "k <> 7", True),
    # Tautology — every row survives.
    (
        "taut",
        lambda: bt.col("i").is_null() | bt.col("i").is_not_null(),
        "i IS NULL OR i IS NOT NULL",
        True,
    ),
    # Partial-overlap ranges / equalities — NOT exact, must fall back (None).
    ("i_gt_2", lambda: bt.col("i") > 2, "i > 2", False),
    ("i_eq_inside", lambda: bt.col("i") == 3, "i = 3", False),
    ("i_between", lambda: bt.col("i").between(2, 4), "i BETWEEN 2 AND 4", False),
]


@pytest.mark.parametrize("name,pred,where,fires", _CASES, ids=[c[0] for c in _CASES])
def test_filter_count_matches_duckdb(pq_path, duck, name, pred, where, fires):
    _duck(duck)
    expected = duck.execute(f"select count(*) from t where {where}").fetchone()[0]

    # Parquet: the metadata shortcut fires (or provably falls back) and, when it fires,
    # equals DuckDB; the executed count always equals DuckDB.
    ds = bt.read.parquet(pq_path).filter(pred())
    answer = _count(ds)
    if fires:
        assert answer == expected, f"{name}: metadata {answer} != duck {expected}"
    else:
        assert answer is None, f"{name}: partial overlap must not answer, got {answer}"
    assert ds.count() == expected

    # In memory: no footer bounds/null counts, so range/null shapes fall back; the
    # executed count still equals DuckDB. (A tautology needs only an exact row count,
    # which an in-memory source has, so it may still fire — and correctly.)
    mem = bt.from_arrow(_TABLE).filter(pred())
    assert mem.count() == expected
    mem_answer = _count(mem)
    assert mem_answer in (None, expected)


def test_is_empty_and_any_match_duckdb(pq_path, duck):
    _duck(duck)
    hub = core.default_hub()
    # A provably-empty filter and a non-empty one, both answered from the footer.
    empty = bt.read.parquet(pq_path).filter(bt.col("i") > 100)
    nonempty = bt.read.parquet(pq_path).filter(bt.col("i").is_not_null())
    assert answer_filter_is_empty(empty._plan, empty._sources, _stats(empty), hub) is True
    assert answer_filter_any(empty._plan, empty._sources, _stats(empty), hub) is False
    assert answer_filter_is_empty(nonempty._plan, nonempty._sources, _stats(nonempty), hub) is False
    assert answer_filter_any(nonempty._plan, nonempty._sources, _stats(nonempty), hub) is True
    # Cross-check the public terminals against DuckDB.
    assert empty.is_empty() is True
    assert empty.has_rows is False
    assert nonempty.has_rows is True


def test_filter_count_through_projection(pq_path, duck):
    # A row-preserving projection above the filter keeps the count exact and firing.
    _duck(duck)
    expected = duck.execute("select count(*) from t where i IS NULL").fetchone()[0]
    ds = bt.read.parquet(pq_path).filter(bt.col("i").is_null()).select("i", "s")
    assert _count(ds) == expected
    assert ds.count() == expected


def test_plain_scan_is_not_a_filter_count(pq_path):
    # No filter → this module declines (that count is `answer_count`'s job), returns None.
    ds = bt.read.parquet(pq_path)
    assert _count(ds) is None
