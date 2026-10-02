"""The adaptive gate stages a query whose one-shot route would hold too much in memory.

The one-shot route streams its largest input and reads every other one whole, so a plan that
scans a fact table twice holds the second copy. TPC-H q18 at sf100 held 600M `lineitem` rows
and was killed at 22 GB on a 30 GiB box; staged, it measures its `HAVING` subquery and runs in
19.3 s. `_holds_too_much` is that check, and it must outrank every cost-based reason to stay
one-shot, since a route that dies has no time to be learned from.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pytest

import batcher as bt
from batcher.api.adaptive import gating
from batcher.config import active_config, config_context

pytestmark = pytest.mark.unit

_ROWS = 50_000


def _self_join() -> bt.Dataset:
    """Two inputs of `_ROWS` rows and two int64 columns each: 800 KB per input projected."""
    left = bt.from_pydict({"k": list(range(_ROWS)), "v": list(range(_ROWS))})
    right = bt.from_pydict({"k": list(range(_ROWS)), "w": list(range(_ROWS))})
    return left.join(right, on="k").select("k", "v", "w")


def _budget(nbytes: int) -> contextlib.AbstractContextManager:
    """A config whose spill budget is `nbytes` (90% of `max_memory_bytes`), or unbounded at 0."""
    cfg = active_config()
    if nbytes == 0:
        memory = dataclasses.replace(cfg.memory, unbounded_memory=True)
    else:
        memory = dataclasses.replace(cfg.memory, max_memory_bytes=int(nbytes / 0.9))
    return config_context(dataclasses.replace(cfg, memory=memory))


def test_a_held_input_past_half_the_budget_is_too_much():
    ds = _self_join()
    with _budget(1_000_000):  # half is 500 KB; the held input projects to ~800 KB
        assert gating._holds_too_much(ds._plan, ds._sources, None) is True


def test_a_held_input_well_inside_the_budget_is_not():
    ds = _self_join()
    with _budget(1_000_000_000):
        assert gating._holds_too_much(ds._plan, ds._sources, None) is False


def test_an_unbounded_budget_never_stages_for_memory():
    ds = _self_join()
    with _budget(0):  # the user opted out of bounded memory
        assert gating._holds_too_much(ds._plan, ds._sources, None) is False


def test_a_single_input_holds_nothing_beside_its_stream():
    ds = bt.from_pydict({"k": list(range(_ROWS))}).group_by("k").agg(n=bt.col("k").count())
    with _budget(1_000):
        assert gating._holds_too_much(ds._plan, ds._sources, None) is False


def test_memory_outranks_every_reason_to_stay_one_shot(monkeypatch):
    """Streams-whole and a learned one-shot route both say stay; holding too much overrides."""
    ds = _self_join()
    monkeypatch.setattr(gating, "_large_enough", lambda *a: True)
    monkeypatch.setattr(gating, "_streams_whole", lambda *a: True)
    monkeypatch.setattr(gating, "_learned_adaptive_route", lambda *a: "one_shot")
    monkeypatch.setattr(gating, "_holds_too_much", lambda *a: True)
    assert gating.resolve_adaptive("auto", ds._plan, ds._sources, None) is True
    # The control: the same plan, holding nothing, stays on the one-shot route.
    monkeypatch.setattr(gating, "_holds_too_much", lambda *a: False)
    assert gating.resolve_adaptive("auto", ds._plan, ds._sources, None) is False
