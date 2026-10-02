"""A user's memory cap binds inside the engine, and every spill path reports that it spilled.

Two defects, pinned together because each hid the other:

**The cap did not bind after the first query.** The engine's memory pool is process-wide and
its limit only grows (`bc_py::process::shared_memory_pool`), and the pool was the engine's
spill authority. So once any query had run under the auto-sensed envelope (tens of GB), a
later query under ``memory.max_memory_bytes = 8 MB`` shared a pool sized for that one: an
aggregate whose state Kyber underestimated -- so Carbonite admitted it in memory -- held
24 MB of state against the 7.2 MB cap and never spilled. It held only for the first query in
the process, which is why a test that ran alone could not see it. The fix makes the query's
own envelope a ceiling in `bc_interp::par::admit`, beside the pool's.

**`spilled` read False for every query that went out of core through the Python executors**
(`dist.spill`, `dist.spill_breakers`), because `QueryProfile.spilled` looked only at engine
operator metrics and that path runs no metered engine call. The fused join-aggregate had the
same defect inside the engine: it spilled and pushed a literal ``false``.

Every assertion reads the engine's or the spill store's own measurement, never timing, and
each has a control proving the reading can come out the other way.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.native import engine
from batcher.config import active_config, config_context

pytestmark = pytest.mark.integration

duckdb = pytest.importorskip("duckdb")

_N = 1_000
#: The cap. The aggregate below holds ~24 MB of state; the engine budget is 90% of this.
_CAP = 8_000_000


def _capped(cap: int = _CAP):
    base = active_config()
    return base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=cap))


def _underestimated_join_aggregate() -> tuple[bt.Dataset, pa.Table, pa.Table]:
    """A join Kyber sizes at ~1,000 rows that produces 1,000,000 distinct groups.

    The join key is computed (`a * 0`), so Kyber has no distinct count for it and estimates
    the join at its inputs' size; every row actually matches every row. Carbonite therefore
    admits the plan in memory, and only the engine can see that its state does not fit.
    """
    left = pa.table({"a": np.arange(_N, dtype="int64")})
    right = pa.table({"b": np.arange(_N, dtype="int64")})
    ds = (
        bt.from_arrow(left)
        .with_columns(j=col("a") * 0)
        .join(bt.from_arrow(right).with_columns(j=col("b") * 0), on="j")
        .group_by("a", "b")
        .agg(n=col("b").count())
    )
    return ds, left, right


def _profile(ds: bt.Dataset) -> dict:
    return json.loads(ds.explain(analyze=True, format="json"))


def _aggregate(profile: dict) -> dict:
    return next(op for op in profile["ops"] if op["kind"] == "aggregate")


@pytest.fixture
def duck():
    """A fresh in-memory DuckDB connection, the oracle for the capped result."""
    con = duckdb.connect()
    yield con
    con.close()


def _grow_the_shared_pool() -> int:
    """Run one uncapped query, as any session does first, and return the pool's limit."""
    bt.from_pydict({"k": [1, 1, 2], "v": [1, 2, 3]}).group_by("k").agg(s=col("v").sum()).collect()
    stats = engine().engine_pool_stats()
    assert stats is not None, "an uncapped query must have created the process pool"
    return int(stats["limit_bytes"])


def test_a_small_cap_spills_in_the_engine_after_an_uncapped_query(duck):
    """The regression: a cap set after the pool grew must still make the engine spill."""
    limit = _grow_the_shared_pool()
    # The precondition the defect needed. Without it this test would pass on the old engine.
    assert limit > _CAP, f"the shared pool ({limit} B) must be larger than the cap"

    ds, left, right = _underestimated_join_aggregate()
    with config_context(_capped()):
        profile = _profile(ds)
        out = ds.collect()

    # Carbonite admitted it: this is the engine's own spill, not the Python out-of-core route.
    assert not any(d["category"] == "spill" for d in profile["decisions"]), profile["decisions"]
    agg = _aggregate(profile)
    assert agg["spilled"], f"1M groups against a {_CAP} B cap stayed resident: {agg}"
    assert agg["spill_bytes"] > 0, agg
    assert profile["spilled"] is True
    duck.register("l", left)
    duck.register("r", right)
    assert_same(out, duck.sql("SELECT a, b, count(b) AS n FROM l, r GROUP BY a, b"))


def test_control_the_same_query_uncapped_stays_resident():
    """The spill above is the cap's doing: without it the same plan does not spill."""
    _grow_the_shared_pool()
    ds, _, _ = _underestimated_join_aggregate()
    profile = _profile(ds)
    assert _aggregate(profile)["spilled"] is False
    assert profile["spilled"] is False


def _table() -> pa.Table:
    rng = np.random.default_rng(0)
    n = 200_000
    return pa.table({"k": rng.integers(0, 100_000, n), "v": rng.integers(0, 100, n)})


#: A sort goes out of core by range-partitioning its input into buckets on disk, whatever the
#: data, so the Python route is certain to write something under a 1 MiB cap.
def _python_spill_sort() -> bt.Dataset:
    return bt.from_arrow(_table()).sort(col("k"))


def _is_sorted(table: pa.Table) -> bool:
    keys = np.asarray(table.column("k"))
    return bool(np.all(keys[:-1] <= keys[1:]))


def test_the_python_out_of_core_path_reports_that_it_spilled():
    """A query Carbonite routes out of core reports `spilled` with the bytes it wrote."""
    ds = _python_spill_sort()
    with config_context(_capped(1 << 20)):
        profile = _profile(ds)
        out = ds.collect()
    # The route this test is about: Carbonite's out-of-core decision, not an engine spill.
    assert any(d["category"] == "spill" for d in profile["decisions"]), profile["decisions"]
    assert not any(op["spilled"] for op in profile["ops"]), "no engine operator ran metered"
    assert profile["spilled"] is True
    assert profile["total_spill_bytes"] > 0
    assert out.num_rows == 200_000 and _is_sorted(out)


def test_control_the_same_query_in_memory_reports_no_spill():
    """The positive control: the identical query, uncapped, stays in memory and says so."""
    profile = _profile(_python_spill_sort())
    assert not any(d["category"] == "spill" for d in profile["decisions"])
    assert profile["spilled"] is False
    assert profile["total_spill_bytes"] == 0


def test_the_out_of_core_route_that_writes_nothing_reports_no_spill():
    """`spilled` is a measurement of what reached disk, not a restatement of the route.

    The out-of-core aggregate holds its partials in memory until they outgrow their budget
    (`dist.spill.aggregate`), and this one never does: Carbonite routed it out of core, and
    nothing was written. Reporting `True` here would be the route-as-fact mistake the
    `record_spill` comment in `api.orchestration.stages` already warns about.
    """
    ds = bt.from_arrow(_table()).group_by("k").agg(s=col("v").sum())
    with config_context(_capped(1 << 20)):
        profile = _profile(ds)
    assert any(d["category"] == "spill" for d in profile["decisions"]), profile["decisions"]
    assert profile["spilled"] is False
    assert profile["total_spill_bytes"] == 0
