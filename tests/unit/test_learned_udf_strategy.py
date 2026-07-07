"""Learned UDF execution policy: the threads-vs-processes verdict and the per-row cost that
sizes the thread batch are persisted to the hub, so a recurring `map_batches` `fn` starts tuned
across runs instead of re-probing every session.

Result-invariance is the headline gate: the verdict and the batch size are *scheduling* choices —
threads vs processes and coarse vs fine batches compute byte-identical output — so a query run with
a seeded (warm) policy must equal the same query run cold.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher.core import default_hub
from batcher.core.udf import strategy as strat

pytestmark = pytest.mark.unit


def _double(b: pa.RecordBatch) -> pa.RecordBatch:
    """A vectorized (module-level, picklable) UDF — cheap per row, so it probes as *light*."""
    return b.append_column("y", pc.multiply(b.column("x"), 2))


def _op_for(fn) -> object:
    """A real `MapBatches` node for `fn` (built via the public API)."""
    t = pa.table({"x": list(range(300_000))})
    return bt.from_arrow(t).ml.map_batches(fn)._plan


def _clear_caches() -> None:
    strat._PROC_PROBE_CACHE.clear()
    strat._FN_ROW_SECONDS.clear()


# --- threads-vs-processes verdict is persisted / seeded ----------------------------------


def test_process_verdict_seeds_from_hub_without_probing(monkeypatch):
    op = _op_for(_double)
    key = strat._fn_probe_key(op.fn)
    assert key is not None
    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {"proc": True})

    # If the persisted verdict is honored, the probe never runs — make it explode to prove it.
    monkeypatch.setattr(
        strat, "_run_proc_probe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed"))
    )
    assert strat._prefer_processes(op, total_rows=2_000_000, current=[]) is True
    assert strat._PROC_PROBE_CACHE[key] is True  # and it seeded the in-process cache


def test_process_verdict_persisted_after_a_cold_probe():
    op = _op_for(_double)
    key = strat._fn_probe_key(op.fn)
    _clear_caches()
    assert default_hub().get_keyed_param(strat._LEARN_NS, key) is None  # cold

    current = pa.table({"x": list(range(300_000))}).to_batches()
    verdict = strat._prefer_processes(op, total_rows=2_000_000, current=current)
    stored = default_hub().get_keyed_param(strat._LEARN_NS, key) or {}
    assert stored.get("proc") == verdict  # the cold probe wrote its verdict back


# --- per-row cost seeds the thread batch size --------------------------------------------


def test_learned_row_cost_changes_the_thread_batch_size():
    op = _op_for(_double)
    key = strat._fn_probe_key(op.fn)
    current = pa.table({"x": list(range(300_000))}).to_batches()
    morsel = 16_384

    # Seeded HEAVY (per-row cost above the light threshold): the batch keeps the per-worker split
    # (floor stays at the morsel), so with many workers the target collapses toward the morsel.
    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {"row_secs": 1.0})
    heavy = strat.thread_batch_target(
        op, 4_000_000, num_workers=256, morsel=morsel, current=current
    )

    # Cold: the trivial `fn` probes as LIGHT, so the floor lifts to the amortization plateau.
    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {})  # explicitly cold
    light = strat.thread_batch_target(
        op, 4_000_000, num_workers=256, morsel=morsel, current=current
    )

    assert heavy < light  # the learned per-row cost steered the batch size
    assert light >= strat._THREAD_MIN_COARSE_ROWS


def test_learned_row_cost_is_persisted_after_measuring():
    op = _op_for(_double)
    key = strat._fn_probe_key(op.fn)
    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {})
    current = pa.table({"x": list(range(300_000))}).to_batches()
    strat.thread_batch_target(op, 4_000_000, num_workers=8, morsel=16_384, current=current)
    stored = default_hub().get_keyed_param(strat._LEARN_NS, key) or {}
    assert isinstance(stored.get("row_secs"), (int, float))  # measured cost written back


# --- result-invariance: a warm vs cold policy gives identical output ---------------------


def test_batch_size_policy_is_result_invariant():
    t = pa.table({"x": list(range(300_000))})
    op = bt.from_arrow(t).ml.map_batches(_double)._plan
    key = strat._fn_probe_key(op.fn)

    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {"row_secs": 1.0})  # heavy → fine batches
    heavy = bt.from_arrow(t).ml.map_batches(_double).to_pydict()

    _clear_caches()
    default_hub().put_keyed_param(strat._LEARN_NS, key, {"row_secs": 1e-12})  # light → coarse
    light = bt.from_arrow(t).ml.map_batches(_double).to_pydict()

    assert heavy == light  # batch size only shards — byte-identical result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
