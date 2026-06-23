"""Bloom data-skipping in `zonemap_prune_filter` — prune equality/IN on an absent
value that min/max cannot rule out, and never prune a present one."""

from __future__ import annotations

import batcher._native as nat
import pyarrow as pa

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.metadata.source_stats_store import _decode, _encode
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

# A column whose values span [1, 10] but skip 7 — so min/max cannot prune `x = 7`,
# only the bloom can.
_VALS = [1, 5, 10]


def _ds_and_stats():
    bloom = nat.build_column_bloom([pa.record_batch({"x": pa.array(_VALS, pa.int64())})], 0, 1000)
    stats = SourceStatistics(
        row_count=len(_VALS),
        columns={
            "x": ColumnStat(
                min=1, max=10, null_count=0, ndv=3, provenance=Provenance.EXACT, bloom=bloom
            )
        },
    )
    ds = bt.from_pydict({"x": _VALS})
    return ds, stats


def _optimize(ds, stats, predicate):
    plan = ds.filter(predicate)._plan
    return Optimizer(sources=ds._sources, source_stats=[stats]).optimize(plan).ir


def _is_pruned(ir) -> bool:
    return ir["op"] == "limit" and ir["n"] == 0


def test_absent_value_in_range_pruned():
    ds, stats = _ds_and_stats()
    # 7 is within [1, 10] (min/max can't help) but absent → the bloom prunes to empty.
    assert _is_pruned(_optimize(ds, stats, col("x") == 7))


def test_present_value_not_pruned():
    ds, stats = _ds_and_stats()
    assert not _is_pruned(_optimize(ds, stats, col("x") == 5))


def test_in_list_all_absent_pruned():
    ds, stats = _ds_and_stats()
    # IN desugars to OR of equalities; all absent (in range) → the whole filter prunes.
    assert _is_pruned(_optimize(ds, stats, col("x").is_in([6, 7, 8])))


def test_in_list_one_present_not_pruned():
    ds, stats = _ds_and_stats()
    assert not _is_pruned(_optimize(ds, stats, col("x").is_in([6, 5, 8])))


def test_no_bloom_falls_back_to_minmax():
    # Without a bloom, an in-range equality is undecidable from min/max → not pruned.
    ds = bt.from_pydict({"x": _VALS})
    stats = SourceStatistics(
        row_count=3, columns={"x": ColumnStat(min=1, max=10, provenance=Provenance.EXACT)}
    )
    assert not _is_pruned(_optimize(ds, stats, col("x") == 7))


def test_source_stats_bloom_round_trips():
    _, stats = _ds_and_stats()
    restored = _decode(_encode(stats))
    assert restored is not None
    assert restored.columns["x"].bloom == stats.columns["x"].bloom
