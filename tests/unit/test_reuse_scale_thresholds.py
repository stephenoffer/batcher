"""Three size thresholds that held at TPC-DS sf1 and broke at sf10, pinned as decisions.

Each of these is a pure decision -- nothing here executes a query -- and each one was a
scale cliff rather than a slow path: the same query crossed a fixed number between one scale
factor and the next and fell onto a much worse plan.

* **The common-subplan budget was a constant.** 256 MiB held q67's shared ROLLUP aggregate at
  sf1 (57 MB) and refused it at sf10 (572 MB), so each of its nine levels recomputed it.
  `_budget_bytes` now rises with the memory budget.
* **The largest repeat was a comma join cut away from its `WHERE`.** q59 references its CTE
  as `FROM wss, store, date_dim` twice, so the largest repeated subtree is `wss x store`, a
  cartesian product as written. It was chosen over `wss` itself and overflowed the budget.
* **The spill gate charged preloaded tables twice.** An in-memory table is already held, and
  an envelope sensed from free RAM was measured with it held; counting it again as input sent
  q47 out of core for memory the spill could not release.

The execution half -- that every rewrite returns what DuckDB returns -- is
`tests/differential/test_diff_reuse_scale_thresholds.py`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.common_subplan as cs
from batcher.api.orchestration.sizing import projected_input_bytes, resident_input_bytes
from batcher.api.subplan_reuse import _budget_bytes, _one_id_per_source
from batcher.carbonite import ResourceManager
from batcher.config import Config, MemoryConfig, config_context
from batcher.plan.ids import OpId
from batcher.plan.logical import Aggregate, Join, Project
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit

_GIB = 1 << 30


class _Estimator:
    """Every node is `rows` rows of 16 bytes, except a global aggregate, which is one row."""

    def __init__(self, rows: float = 1000.0):
        self._rows = rows

    def estimate(self, node):
        while isinstance(node, Project):
            node = node.input
        one = isinstance(node, Aggregate) and not node.group_keys
        return type("Stats", (), {"rows": 1.0 if one else self._rows})()

    def row_width(self, _node, _default):
        return 16.0


def _session() -> bt.Session:
    rng = np.random.default_rng(0)
    n = 2_000
    s = bt.Session()
    s.register(
        "fact",
        pa.table({"k": rng.integers(0, 300, n), "s": rng.integers(0, 20, n), "v": rng.random(n)}),
    )
    s.register("dim", pa.table({"dk": np.arange(300), "dw": np.arange(300) // 7}))
    s.register("st", pa.table({"sk": np.arange(20), "name": [f"s{i}" for i in range(20)]}))
    return s


# The TPC-DS q59 shape: a CTE referenced twice, each time as the head of a comma join whose
# real condition (`s = sk`) is in the WHERE clause above the whole FROM list.
_COMMA_CTE = """
WITH w AS (SELECT dw, s, sum(v) tot FROM fact, dim WHERE dk = k GROUP BY dw, s)
SELECT a.name, a.dw, a.tot / b.tot r FROM
  (SELECT name, w.dw, tot FROM w, st, dim d WHERE d.dw = w.dw AND s = sk AND d.dk < 100) a,
  (SELECT name, w.dw, tot FROM w, st, dim d WHERE d.dw = w.dw AND s = sk AND d.dk >= 100) b
WHERE a.name = b.name AND a.dw = b.dw - 5
"""


def _candidates(sql: str):
    ds = _session().sql(sql)
    plan = _one_id_per_source(ds._plan, ds._sources)
    return cs.common_subplans(plan, lambda: _Estimator(), max_bytes=256 << 20, row_bytes=64)


def _has_cartesian_join(node) -> bool:
    return any(isinstance(n, Join) and cs._is_cartesian(n) for n in walk(node))


def test_a_repeat_cut_from_its_where_shares_the_aggregate_inside_it():
    found = _candidates(_COMMA_CTE)
    assert len(found) == 1
    aggregates = [n for n in walk(found[0]) if isinstance(n, Aggregate)]
    assert aggregates, "the CTE's aggregate is what repeats, and what must be shared"
    assert not cs._pending_cartesian(found[0], _Estimator()), (
        "the shared subtree must not be a cartesian product whose condition was left above it"
    )


def test_without_the_check_the_cartesian_repeat_is_what_gets_chosen(monkeypatch):
    """The positive control: the shape really does offer the cross product first."""
    pending = cs._pending_cartesian
    monkeypatch.setattr(cs, "_pending_cartesian", lambda node, estimator: False)
    found = _candidates(_COMMA_CTE)
    assert len(found) == 1
    assert pending(found[0], _Estimator()), "the unchecked choice is the cut-away comma join"


def test_a_comma_join_under_its_own_where_is_not_pending():
    """`wss` is itself `FROM fact, dim WHERE dk = k`: its filter travels with it."""
    found = _candidates(_COMMA_CTE)
    assert _has_cartesian_join(found[0]), "the CTE does hold a pseudo-join under its WHERE"
    assert not cs._pending_cartesian(found[0], _Estimator())


def test_a_scalar_cross_join_is_not_pending():
    """A pseudo-join against a one-row side is a scalar broadcast, not a product to avoid."""
    ds = _session().sql(
        "SELECT a.k, a.t / b.total r FROM (SELECT k, sum(v) t FROM fact GROUP BY k) a, "
        "(SELECT sum(v) total FROM fact) b"
    )
    joins = [n for n in walk(ds._plan) if isinstance(n, Join) and cs._is_cartesian(n)]
    assert joins, "the query must lower to a cartesian pseudo-join for this to test anything"
    assert not cs._pending_cartesian(joins[0], _Estimator())


def test_a_cross_join_of_two_multi_row_sides_is_pending():
    """The same pseudo-join with no `WHERE` above it, and neither side a single row."""
    ds = _session().sql(
        "SELECT a.k, b.k2 FROM (SELECT k, sum(v) t FROM fact GROUP BY k) a, "
        "(SELECT s k2, sum(v) t2 FROM fact GROUP BY s) b"
    )
    joins = [n for n in walk(ds._plan) if isinstance(n, Join) and cs._is_cartesian(n)]
    assert joins
    assert cs._pending_cartesian(joins[0], _Estimator())


# --- the reuse budget ------------------------------------------------------------------


def _config(envelope: int | None, *, cap: int = 256 << 20, fraction: float = 1 / 16) -> Config:
    cfg = Config()
    return cfg.replace(
        memory=dataclasses.replace(cfg.memory, max_memory_bytes=envelope),
        optimizer=dataclasses.replace(
            cfg.optimizer, common_subplan_max_bytes=cap, common_subplan_memory_fraction=fraction
        ),
    )


def test_the_budget_rises_with_the_memory_budget():
    small, large = _budget_bytes(_config(16 * _GIB)), _budget_bytes(_config(256 * _GIB))
    assert small < large
    assert large > 572_000_000, "q67's 572 MB shared aggregate fits a large machine's budget"


def test_the_budget_never_falls_below_the_configured_cap():
    assert _budget_bytes(_config(1 * _GIB)) == 256 << 20


def test_the_share_is_a_power_of_two_so_free_ram_jitter_does_not_rekey():
    a = _budget_bytes(_config(100 * _GIB))
    b = _budget_bytes(_config(100 * _GIB + (300 << 20)))
    assert a == b
    assert a & (a - 1) == 0


def test_a_zero_fraction_keeps_the_cap_fixed():
    assert _budget_bytes(_config(256 * _GIB, fraction=0.0)) == 256 << 20


def test_a_zero_cap_still_turns_reuse_off():
    assert _budget_bytes(_config(256 * _GIB, cap=0)) == 0


# --- resident input under the spill gate --------------------------------------------------


def _held(envelope: int, *, sensed: bool):
    return Config().replace(
        memory=MemoryConfig(max_memory_bytes=envelope, max_memory_bytes_sensed=sensed)
    )


def _sources(tmp_path):
    table = pa.table({"x": np.arange(10_000), "y": np.arange(10_000) * 2})
    resident = bt.from_arrow(table)
    path = str(tmp_path / "t.parquet")
    resident.write.parquet(path)
    on_disk = bt.read.parquet(path)
    return resident._sources + on_disk._sources


def test_a_resident_source_counts_as_held_under_a_sensed_envelope(tmp_path):
    sources = _sources(tmp_path)
    with config_context(_held(_GIB, sensed=True)):
        held = resident_input_bytes(sources, {})
    assert held == projected_input_bytes(sources, {}, [0]) > 0, (
        "only the in-memory source is held; the Parquet file still has to be read"
    )


def test_nothing_counts_as_held_under_a_configured_envelope(tmp_path):
    """A cap the caller set bounds the whole process, preloaded tables included."""
    sources = _sources(tmp_path)
    with config_context(_held(_GIB, sensed=False)):
        assert resident_input_bytes(sources, {}) == 0


def test_only_the_scanned_sources_count(tmp_path):
    sources = _sources(tmp_path)
    with config_context(_held(_GIB, sensed=True)):
        assert resident_input_bytes(sources, {}, scanned={1}) == 0


def _plan(peak_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(0),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=peak_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=float(peak_bytes)),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def test_held_input_is_not_charged_against_the_envelope():
    envelope = 1_000_000
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=envelope))):
        rm = ResourceManager()
        input_bytes, plan = int(envelope * 0.7), _plan(int(envelope * 0.2))
        assert rm.resident_total_exceeds_budget(input_bytes, plan) is True
        assert rm.resident_total_exceeds_budget(input_bytes, plan, held_bytes=input_bytes) is False


def test_held_input_still_leaves_the_state_term_in_force():
    """Excusing the input must not excuse a breaker whose own state does not fit."""
    envelope = 1_000_000
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=envelope))):
        rm = ResourceManager()
        input_bytes, plan = int(envelope * 0.3), _plan(int(envelope * 0.6))
        assert rm.resident_total_exceeds_budget(input_bytes, plan, held_bytes=input_bytes) is True
