"""Learned credit-window sizing — a recurring shuffle warm-starts at its learned window.

A shuffle channel's AIMD credit window converges to a value that balances pipeline
depth against buffered memory. Re-climbing to it from `default_credits` on every run
wastes the early rounds. These tests cover persisting a channel's converged window per
signature and warm-starting `grant_credits` / the AIMD controller from it — a credit
window only bounds in-flight buffering, so it never changes the result (proven by
running a real shuffle query at two windows and asserting an identical result).
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.policies import (
    credit_ceiling,
    load_shuffle_window,
    record_shuffle_window,
)
from batcher.config import active_config
from batcher.config.config import FlowControlConfig
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


# --- persistence -------------------------------------------------------------


def test_record_and_load_shuffle_window():
    hub = _hub()
    assert load_shuffle_window(hub, "shufA") is None  # cold
    record_shuffle_window(hub, "shufA", 40)
    assert load_shuffle_window(hub, "shufA") == 40


def test_shuffle_window_smoothed_across_runs():
    hub = _hub()
    record_shuffle_window(hub, "s", 40)  # first: 40
    record_shuffle_window(hub, "s", 20)  # alpha 0.5 → round(0.5*20 + 0.5*40) = 30
    assert load_shuffle_window(hub, "s") == 30


def test_record_is_best_effort_on_none_hub():
    record_shuffle_window(None, "s", 40)  # must not raise
    assert load_shuffle_window(None, "s") is None


# --- grant_credits / adaptive warm-start ------------------------------------


def test_grant_credits_uses_learned_window():
    hub = _hub()
    record_shuffle_window(hub, "s", 40)
    rm = ResourceManager(hub=hub)
    # Cold path (no signature): the static default.
    assert rm.grant_credits(0) == active_config().flow_control.default_credits
    # Warm path: the learned window, clamped to the memory-safe ceiling.
    assert rm.grant_credits(0, signature="s") == min(40, credit_ceiling(active_config()))


def test_grant_credits_cold_signature_falls_back():
    rm = ResourceManager(hub=_hub())
    assert (
        rm.grant_credits(0, signature="never-seen") == active_config().flow_control.default_credits
    )


def test_adaptive_flow_control_warm_starts_from_learned():
    hub = _hub()
    record_shuffle_window(hub, "s", 30)
    rm = ResourceManager(hub=hub)
    assert rm.adaptive_flow_control(signature="s").window == min(
        30, credit_ceiling(active_config())
    )
    # Cold: the configured default start.
    assert rm.adaptive_flow_control().window == active_config().flow_control.default_credits


def test_learned_window_clamped_to_ceiling():
    hub = _hub()
    record_shuffle_window(hub, "s", 10_000_000)  # absurd
    rm = ResourceManager(hub=hub)
    assert rm.grant_credits(0, signature="s") == credit_ceiling(active_config())


# --- result-invariance -------------------------------------------------------


def _rows(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_credit_window_is_result_invariant():
    # A shuffling aggregate at a tiny (1) vs wide (64) credit window must be identical —
    # the window only bounds in-flight batches, never the computed result.
    t = pa.table({"k": [i % 13 for i in range(5000)], "v": list(range(5000))})

    def q() -> pa.Table:
        return bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=col("v").count()).collect()

    base = active_config()
    with config_context(Config().replace(flow_control=FlowControlConfig(default_credits=1))):
        narrow = q()
    with config_context(Config().replace(flow_control=FlowControlConfig(default_credits=64))):
        wide = q()
    assert _rows(narrow) == _rows(wide) == _rows(q())
    assert base is active_config()  # scope restored
