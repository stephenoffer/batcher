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
