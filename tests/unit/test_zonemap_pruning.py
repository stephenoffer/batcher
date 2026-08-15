"""Zone-map predicate pruning: provably-empty/always-true filters are rewritten.

Plan-shape tests (the rewrite fires) plus the safety boundary (nothing is pruned
without proving bounds, and a column with nulls is never declared always-true).
Result-correctness vs DuckDB is covered in the differential suite.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, lit
from batcher.io.source import source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Filter, Limit


def _optimize(ds):
    stats = [source_statistics(s) for s in ds._sources]
    opt = Optimizer(sources=ds._sources, source_stats=stats)
    return opt.logical_rewrite(ds._plan)


def _pq(tmp_path, table, name="t.parquet"):
    import pyarrow.parquet as pq

    path = str(tmp_path / name)
    pq.write_table(table, path)
    return bt.read.parquet(path)


def test_always_false_filter_becomes_empty(tmp_path):
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter(col("age") < lit(0)))
    # No row can satisfy age < 0 (min age is 18) → rewritten to a zero-row Limit.
    assert isinstance(pruned, Limit) and pruned.n == 0


def test_always_true_filter_is_dropped(tmp_path):
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter(col("age") < lit(1000)))
    # Every age < 1000 and the column has no nulls → the filter is dead.
    assert not isinstance(pruned, (Filter, Limit))


def test_satisfiable_filter_is_untouched(tmp_path):
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter(col("age") < lit(40)))
    # Some rows pass, some don't → undecidable from bounds → left as a Filter.
    assert isinstance(pruned, Filter)


def test_equality_outside_range_is_empty(tmp_path):
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter(col("age") == lit(200)))
    assert isinstance(pruned, Limit) and pruned.n == 0


def test_conjunction_with_empty_conjunct_is_empty(tmp_path):
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter((col("age") > lit(20)) & (col("age") > lit(500))))
    assert isinstance(pruned, Limit) and pruned.n == 0


def test_no_pruning_without_source_stats():
    # In-memory source exposes no min/max bounds → never pruned (executed).
    ds = bt.from_pydict({"age": list(range(18, 80))}).filter(col("age") < lit(0))
    opt = Optimizer(sources=ds._sources)  # no source_stats
    assert isinstance(opt.logical_rewrite(ds._plan), Filter)


# --- all-null columns -----------------------------------------------------------
#
# The mirror of the null-free case, and the one that was undecidable: an EXACT null count
# equal to the row count proves `IS NULL` keeps every row and `IS NOT NULL` keeps none.
# The join rules already proved exactly this about a key (`evidence._all_null_key`), so a
# `WHERE col IS NOT NULL` over such a column was scanning the relation to return nothing.


def _all_null(tmp_path, name="an.parquet"):
    """A Parquet source whose `note` column holds nothing but nulls."""
    import pyarrow.parquet as pq

    path = str(tmp_path / name)
    pq.write_table(
        pa.table({"a": list(range(10)), "note": pa.array([None] * 10, type=pa.string())}), path
    )
    return bt.read.parquet(path)


def test_is_not_null_over_all_null_column_is_empty(tmp_path):
    pruned = _optimize(_all_null(tmp_path).filter(col("note").is_not_null()))
    assert isinstance(pruned, Limit) and pruned.n == 0


def test_is_null_over_all_null_column_is_dropped(tmp_path):
    pruned = _optimize(_all_null(tmp_path).filter(col("note").is_null()))
    # Every row is null, so the filter keeps all of them and is dead.
    assert not isinstance(pruned, (Filter, Limit))


def test_all_null_rewrites_agree_with_execution(tmp_path):
    """The rewrites above delete and keep rows, so pin them against the executed answer."""
    ds = _all_null(tmp_path)
    assert ds.filter(col("note").is_not_null()).collect().num_rows == 0
    assert ds.filter(col("note").is_null()).collect().num_rows == 10


def test_partially_null_column_is_undecidable(tmp_path):
    """A column with *some* nulls decides neither direction and stays a Filter."""
    import pyarrow.parquet as pq

    path = str(tmp_path / "pn.parquet")
    pq.write_table(pa.table({"note": pa.array(["x", None, "y"], type=pa.string())}), path)
    ds = bt.read.parquet(path)
    assert isinstance(_optimize(ds.filter(col("note").is_not_null())), Filter)
    assert isinstance(_optimize(ds.filter(col("note").is_null())), Filter)


# --- IN lists in the shared oracle -----------------------------------------------
#
# `prune_in_list_by_zonemap` already narrows a list member by member, but only for an
# `InList` sitting as a top-level conjunct of a `Filter`. Teaching `_predicate_status`
# the shape reaches the other consumers of the same oracle — a refuted `IN` as one arm
# of an `OR`, and an `IN` constraining a join side.


def test_refuted_in_list_disjunct_is_dropped(tmp_path):
    """`a > 70 OR a IN (<all out of range>)` collapses to the bare sargable comparison."""
    ds = _pq(tmp_path, pa.table({"age": list(range(18, 80))}))
    pruned = _optimize(ds.filter((col("age") > lit(70)) | col("age").is_in([200, 300])))
    assert isinstance(pruned, Filter)
    assert "is_in" not in repr(pruned.predicate) and "200" not in repr(pruned.predicate)


def test_in_list_over_a_constant_column_is_always_true(tmp_path):
    """A constant column whose value the list contains keeps every row — filter dead.

    Needs an EXACT bundle, which a numeric footer bound is and a *string* one is not
    (Parquet may truncate it), so this deliberately uses an integer column.
    """
    ds = _pq(tmp_path, pa.table({"v": [7] * 50}), name="c.parquet")
    pruned = _optimize(ds.filter(col("v").is_in([7, 9])))
    assert not isinstance(pruned, (Filter, Limit))


def test_in_list_missing_the_constant_is_empty(tmp_path):
    ds = _pq(tmp_path, pa.table({"v": [7] * 50}), name="c2.parquet")
    pruned = _optimize(ds.filter(col("v").is_in([1, 9])))
    assert isinstance(pruned, Limit) and pruned.n == 0


def test_in_list_rewrites_agree_with_execution(tmp_path):
    """All three shapes above change which rows come back, so pin them to the answer."""
    ds = _pq(tmp_path, pa.table({"v": [7] * 50}), name="c3.parquet")
    assert ds.filter(col("v").is_in([7, 9])).collect().num_rows == 50
    assert ds.filter(col("v").is_in([1, 9])).collect().num_rows == 0
    ages = _pq(tmp_path, pa.table({"age": list(range(18, 80))}), name="c4.parquet")
    both = ages.filter((col("age") > lit(70)) | col("age").is_in([200, 300]))
    assert both.collect().num_rows == 9


def test_in_list_over_a_nullable_constant_column_is_undecidable(tmp_path):
    """A NULL row makes `IN` evaluate to NULL, which the filter drops — so the filter is
    doing real work and must not be called always-true."""
    import pyarrow.parquet as pq

    path = str(tmp_path / "cn.parquet")
    pq.write_table(pa.table({"v": pa.array([7, 7, None], pa.int64())}), path)
    ds = bt.read.parquet(path)
    assert isinstance(_optimize(ds.filter(col("v").is_in([7, 9]))), Filter)
    assert ds.filter(col("v").is_in([7, 9])).collect().num_rows == 2
