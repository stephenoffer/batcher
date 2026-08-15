"""Statistics Batcher remembered from writing a path reach the optimizer on reading it.

Two column facets exist only because a *write* measured them, and no reader can recover
either: the membership bloom (`optimizer.build_bloom_index`), which refutes a point lookup
sitting inside `[min, max]` where a zone map is blind, and an HLL distinct count, which no
CSV or JSON header carries. `persist_written_source_stats` writes both; `collect_source_stats`
is the only thing that can bring them back.

It brings them back by **merging** them into whatever the source declares for itself, and
that is the behaviour these tests pin. The previous spelling was a fallback — used only
when the source declared nothing at all — which quietly stopped firing once the `FileSource`
base gained a generic `statistics()`: every file source then declared at least a byte size,
so the branch became unreachable and the index was built on every write and read on none.
A test that only checked "the query returns the right rows" could not see that, because an
unconsulted index changes no result. So these assert the *plan shape* and the *facets*, not
the answer.

The version gate is the other half and is a correctness property rather than a performance
one. A bloom proves *absence*, so one describing a previous version of a path deletes rows
that now exist. `test_external_rewrite_*` is the case that must never regress.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import Config, col, config_context, core
from batcher.api.source_stats import _SOURCE_STATS_CACHE, collect_source_stats
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Filter, Limit

pytestmark = pytest.mark.unit

#: Even ids only, so every odd id in `[0, 19998]` is absent *inside* the bounds — the case
#: min/max cannot refute and a bloom can. This is the whole point of the index.
_IDS = list(range(0, 20000, 2))


def _indexed() -> Config:
    """A config that builds the membership index on write (it is off by default)."""
    base = Config()
    return base.replace(optimizer=dataclasses.replace(base.optimizer, build_bloom_index=True))


def _write(path: str, fmt: str, ids: list[int] | None = None) -> None:
    table = pa.table({"id": pa.array(_IDS if ids is None else ids, pa.int64())})
    with config_context(_indexed()):
        getattr(bt.from_arrow(table).write, fmt)(path)


def _stats(ds):
    return collect_source_stats(ds._sources, core.default_hub())


def _optimized(ds):
    return Optimizer(sources=ds._sources, source_stats=_stats(ds)).logical_rewrite(ds._plan)


@pytest.fixture(autouse=True)
def _cold_session_cache():
    """Each test starts with an empty session memo, so a stale entry cannot mask a miss."""
    _SOURCE_STATS_CACHE.clear()
    yield
    _SOURCE_STATS_CACHE.clear()


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_remembered_facets_reach_the_estimator(tmp_path, fmt) -> None:
    """The bloom and the distinct count survive the round trip, for a footer format and
    a footerless one alike.

    Parquet is the case that regressed silently: it declares its own footer statistics, so
    it never took the "source declared nothing" fallback, and the merge is what puts the
    remembered facets beside the footer's bounds instead of discarding them.
    """
    path = str(tmp_path / f"t.{fmt}")
    _write(path, fmt)
    stat = _stats(getattr(bt.read, fmt)(path))[0].columns["id"]
    assert stat.bloom is not None
    assert stat.ndv is not None


def test_live_bounds_are_not_overwritten(tmp_path) -> None:
    """The source stays authoritative for what it knows; the entry adds only what it can't.

    A remembered `min`/`max` is a bound on a *previous* version of the path, and a bound is
    what a prune is decided from — so the merge must never contribute one, even when the
    entry happens to carry it.
    """
    path = str(tmp_path / "t.parquet")
    _write(path, "parquet")
    stat = _stats(bt.read.parquet(path))[0].columns["id"]
    assert (stat.min, stat.max) == (0, 19998)  # the footer's, not the entry's


def test_remembered_ndv_can_never_answer_an_exact_query(tmp_path) -> None:
    """An HLL count is an estimate however exact the bundle it lands beside is.

    It rides into a Parquet column stat whose bounds are EXACT, which is exactly the
    situation `ndv_provenance` exists for: without its own tag it would inherit the
    bundle's and let an approximate count answer `count_distinct`.
    """
    path = str(tmp_path / "t.parquet")
    _write(path, "parquet")
    stat = _stats(bt.read.parquet(path))[0].columns["id"]
    assert not stat.ndv_is_exact


def test_absent_in_range_value_is_pruned_from_the_plan(tmp_path) -> None:
    """The consequence the facets exist for: the scan is removed, not merely correct.

    Asserting the row count here would pass whether or not the index was consulted — 7777
    is genuinely absent — so this asserts the rewritten plan instead.
    """
    path = str(tmp_path / "t.parquet")
    _write(path, "parquet")
    plan = _optimized(bt.read.parquet(path).filter(col("id") == 7777))
    assert isinstance(plan, Limit) and plan.n == 0


def test_present_value_is_not_pruned(tmp_path) -> None:
    """The other side of the same claim: a value the column holds survives as a Filter."""
    path = str(tmp_path / "t.parquet")
    _write(path, "parquet")
    plan = _optimized(bt.read.parquet(path).filter(col("id") == 7778))
    assert isinstance(plan, Filter)


def test_external_rewrite_discards_the_remembered_bloom(tmp_path) -> None:
    """A path rewritten by someone else must not be pruned against the old index.

    This is the failure the version gate exists to prevent, and it is a wrong *answer*
    rather than a slow plan: the row is present after the rewrite, and an absence proof
    carried over from the previous contents would delete it. `invalidate_source_stats`
    covers Batcher's own writes and cannot see this one.
    """
    path = str(tmp_path / "t.parquet")
    _write(path, "parquet")
    _SOURCE_STATS_CACHE.clear()
    # A writer Batcher knows nothing about replaces the file with one holding 7777.
    pq.write_table(pa.table({"id": pa.array([7777], pa.int64())}), path)

    assert bt.read.parquet(path).filter(col("id") == 7777).collect().num_rows == 1


def test_index_off_leaves_no_bloom(tmp_path) -> None:
    """With the index disabled nothing is persisted to merge, and reads stay correct."""
    path = str(tmp_path / "t.parquet")
    table = pa.table({"id": pa.array(_IDS, pa.int64())})
    bt.from_arrow(table).write.parquet(path)  # default config: index off
    stat = _stats(bt.read.parquet(path))[0].columns["id"]
    assert stat.bloom is None
    assert isinstance(_optimized(bt.read.parquet(path).filter(col("id") == 7777)), Filter)
