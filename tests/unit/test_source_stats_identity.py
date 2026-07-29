"""A source's learned statistics must not be readable by an unrelated later source.

`source_stats_key` keys a shape-identified source (in-memory batches, whose `identity()` is
schema + row count) by object identity rather than by that shape, precisely so two different
relations of the same shape cannot share a statistics slot. It used `id()` to do it, and
CPython hands the next object the address of the one just freed — so four in-memory sources
created in sequence produced **one** key between them, each reading and overwriting the
others' distinct counts, most-common-values and quantile grids.

Nothing failed, because a statistic never changes a result. The plans were simply built from
another relation's data.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.source_stats import source_stats_key

pytestmark = pytest.mark.unit


def _frame(fill: int, rows: int = 64):
    return bt.from_arrow(pa.table({"k": pa.array(np.full(rows, fill, dtype="int64"))}))


def test_transient_sources_do_not_share_a_key():
    """The reproduction: build and drop, so each new source lands on a freed address."""
    keys = []
    for i in range(6):
        ds = _frame(i)
        keys.append(source_stats_key(ds._sources[0]))
        del ds
    assert len(set(keys)) == len(keys), keys


def test_two_live_sources_of_identical_shape_differ():
    a, b = _frame(1), _frame(2)
    assert a._sources[0].identity() == b._sources[0].identity()  # the shape really matches
    assert source_stats_key(a._sources[0]) != source_stats_key(b._sources[0])


def test_the_key_is_stable_for_one_source():
    """A serial is allocated once and reused, or every query would re-learn from scratch."""
    ds = _frame(3)
    source = ds._sources[0]
    assert source_stats_key(source) == source_stats_key(source)


def test_a_path_keyed_source_is_unchanged(tmp_path):
    """A file source keys by its path, so what one run measures the next run reads back.

    The serial is only for sources whose identity is shape-based; introducing it must not
    make a Parquet directory's statistics process-local.
    """
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"k": pa.array([1, 2, 3])}), tmp_path / "t.parquet")
    first = bt.read.parquet(str(tmp_path / "t.parquet"))._sources[0]
    second = bt.read.parquet(str(tmp_path / "t.parquet"))._sources[0]
    assert first is not second
    key = source_stats_key(first)
    assert key == source_stats_key(second)
    assert "obj:" not in key


def test_column_statistics_do_not_leak_between_frames():
    """End to end: the statistic itself, not just the key.

    Frame A's column is all 7s and frame B's is all 999s. With one shared key, B's seeded
    distinct count and most-common-values were A's.
    """
    from batcher import core
    from batcher.api.terminal._metadata import seed_column_ndv
    from batcher.kyber import NDV_KEY, columns_for, load_learned_stats

    hub = core.default_hub()
    a, b = _frame(7), _frame(999)
    for ds in (a, b):
        plan = ds.filter(bt.col("k") == 7)._plan
        seed_column_ndv(hub, ds._sources, plan)
    learned = load_learned_stats(hub)
    key_a = source_stats_key(a._sources[0])
    key_b = source_stats_key(b._sources[0])
    assert key_a != key_b
    # Both are constant columns, so both read ndv ~1 under their *own* key. (The sketch is
    # HyperLogLog, so the count is approximate by construction — hence the tolerance.)
    assert columns_for(learned, NDV_KEY, key_a).get("k") == pytest.approx(1, rel=0.01)
    assert columns_for(learned, NDV_KEY, key_b).get("k") == pytest.approx(1, rel=0.01)
