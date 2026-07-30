"""What the profile says about spilling must be what actually happened.

`spill_to_disk` asks the out-of-core executor to run the plan, and gets `None` back when
the shape has no spilling path — a string-keyed sort, or a filter/project with no state to
spill. The caller then falls through to the in-memory path, which is the intended
behaviour.

What was not intended is that the profile had already been told a spill happened. The
`record_spill` call sat *before* `spill_collect`, so a run that stayed entirely resident
still reported ``carbonite_summary = "out-of-core spill (N partitions)"`` and a Carbonite
decision reading "executed out-of-core under bounded memory". That is the single claim a
reader reaches for while diagnosing an OOM, and it was backwards precisely on the runs that
never spilled.

No existing test could see it: `tests/integration/test_spilling.py` asserts only that the
spilled result equals the in-memory one, which is trivially true when the "spilled" run
*was* the in-memory one. The profile is the observable that distinguishes them, so it is
what these tests assert on.
"""

from __future__ import annotations

import types

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, core, kyber
from batcher.api.orchestration import stages
from batcher.plan.physical import PhysicalPlan
from batcher.plan.profile import ProfileCollector

pytestmark = pytest.mark.unit


class _RM:
    """A resource manager stub exposing only what `spill_to_disk` asks of it."""

    def __init__(self, partitions: int = 8, reason: str = "input exceeds the envelope"):
        self._partitions = partitions
        self._reason = reason

    def recommend_spill_partitions(self, opt):
        return self._partitions

    def partitions_for_bounds(self, opt, bounds):
        return 0

    def spill_reason(self, opt):
        return self._reason

    def __getattr__(self, name):
        # `spill_compression_scope` consults the manager for the learned codec; every such
        # query answering `None` selects the default, which is what an unmeasured plan gets.
        return lambda *args, **kwargs: None


def _plan():
    ds = bt.from_arrow(pa.table({"v": [3, 1, 2], "s": ["c", "a", "b"]})).sort(col("s"))
    return (
        kyber.optimize_logical(ds._plan, sources=ds._sources, hub=core.default_hub()),
        ds._sources,
    )


def _run(monkeypatch, result):
    """Call `spill_to_disk` with `spill_collect` forced to return `result`.

    Patching the executor is what makes both outcomes reachable as a unit test: whether a
    given shape has an out-of-core path is a property of the engine, and this file is about
    what the profile reports for each outcome, not about which shapes have one.
    """
    import batcher.dist.spill as spill_module

    monkeypatch.setattr(spill_module, "spill_collect", lambda *a, **k: result)
    logical_opt, sources = _plan()
    ctx = types.SimpleNamespace(profile=ProfileCollector())
    verdict = types.SimpleNamespace(suggested_bounds=None, advisory=True)
    out = stages.spill_to_disk(
        logical_opt, sources, ctx, _RM(), PhysicalPlan(ir={}, output_schema=None, ops=()), verdict
    )
    return out, ctx.profile


def _spill_decisions(profile) -> list:
    return [d for d in (profile.decisions or []) if d.category == "spill"]


def test_a_run_that_fell_back_to_memory_does_not_claim_it_spilled(monkeypatch):
    """The regression: `None` back from the executor means nothing was spilled."""
    out, profile = _run(monkeypatch, None)
    assert out is None, "the stub forces the no-out-of-core-path outcome"
    assert not _spill_decisions(profile), (
        "a run with no out-of-core path recorded a spill decision: "
        f"{[d.summary for d in _spill_decisions(profile)]}"
    )
    summary = getattr(profile, "carbonite_summary", None)
    assert not (summary and "out-of-core" in summary), (
        f"carbonite_summary claims an out-of-core run that did not happen: {summary!r}"
    )


def test_a_run_that_actually_spilled_still_reports_it(monkeypatch):
    """...and the reporting is not simply removed — the true case must stay true.

    Without this, deleting the `record_spill` call would satisfy the test above and lose
    the signal entirely, which is a worse outcome than the bug it fixes.
    """
    table = pa.table({"v": [1], "s": ["a"]})
    out, profile = _run(monkeypatch, table)
    assert out is table
    decisions = _spill_decisions(profile)
    assert len(decisions) == 1, f"expected one spill decision, got {decisions}"
    assert decisions[0].subsystem == "carbonite", (
        "spilling is a Carbonite decision — it protects, it does not optimize"
    )
    assert "out-of-core" in (profile.carbonite_summary or "")


def test_the_recorded_spill_carries_the_partition_count_and_reason(monkeypatch):
    """The numbers in the profile are the ones the run used, not placeholders."""
    import batcher.dist.spill as spill_module

    table = pa.table({"v": [1], "s": ["a"]})
    monkeypatch.setattr(spill_module, "spill_collect", lambda *a, **k: table)
    logical_opt, sources = _plan()
    ctx = types.SimpleNamespace(profile=ProfileCollector())
    verdict = types.SimpleNamespace(suggested_bounds=None, advisory=True)
    stages.spill_to_disk(
        logical_opt,
        sources,
        ctx,
        _RM(partitions=32, reason="learned peak exceeds the envelope"),
        PhysicalPlan(ir={}, output_schema=None, ops=()),
        verdict,
    )
    decision = _spill_decisions(ctx.profile)[0]
    assert decision.detail["partitions"] == 32
    assert decision.detail["reason"] == "learned peak exceeds the envelope"
    assert "32" in ctx.profile.carbonite_summary


def test_the_partition_count_the_profile_reports_is_the_one_passed_to_the_executor(monkeypatch):
    """A profile that reports a different fan-out than the run used is a subtler lie than
    reporting a run that never happened, and just as misleading when sizing a machine."""
    import batcher.dist.spill as spill_module

    seen: dict[str, int] = {}
    table = pa.table({"v": [1], "s": ["a"]})

    def _capture(logical, sources, partitions):
        seen["partitions"] = partitions
        return table

    monkeypatch.setattr(spill_module, "spill_collect", _capture)
    logical_opt, sources = _plan()
    ctx = types.SimpleNamespace(profile=ProfileCollector())
    verdict = types.SimpleNamespace(suggested_bounds=None, advisory=True)
    stages.spill_to_disk(
        logical_opt,
        sources,
        ctx,
        _RM(partitions=17),
        PhysicalPlan(ir={}, output_schema=None, ops=()),
        verdict,
    )
    assert seen["partitions"] == 17
    assert _spill_decisions(ctx.profile)[0].detail["partitions"] == seen["partitions"]


def test_no_profile_attached_is_not_an_error(monkeypatch):
    """Profiling is optional; the spill path must run identically without a collector."""
    import batcher.dist.spill as spill_module

    table = pa.table({"v": [1], "s": ["a"]})
    monkeypatch.setattr(spill_module, "spill_collect", lambda *a, **k: table)
    logical_opt, sources = _plan()
    ctx = types.SimpleNamespace(profile=None)
    verdict = types.SimpleNamespace(suggested_bounds=None, advisory=True)
    out = stages.spill_to_disk(
        logical_opt, sources, ctx, _RM(), PhysicalPlan(ir={}, output_schema=None, ops=()), verdict
    )
    assert out is table
