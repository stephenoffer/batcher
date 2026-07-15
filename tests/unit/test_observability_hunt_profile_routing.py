"""Observability hunt: the profiled path (`stats()` / `explain(analyze=True)`) must route
exactly as `collect()` does, so the profile it measures reflects the real query."""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.api.terminal.core as core_mod

pytestmark = pytest.mark.unit


def test_run_profiled_routes_with_plan_and_sources(monkeypatch):
    """`run_profiled` must forward plan + sources to the `distributed="auto"` resolver.

    Regression: it called `resolve_distributed("auto")` with neither, hitting the
    `sources is None -> return True` fall-through, so on a multi-node Ray cluster
    `stats()`/`explain(analyze=True)` forced *every* query to distribute — even a tiny
    one that `collect()` runs single-node — measuring a path the real query never takes.
    """
    orig = core_mod._resolve_distributed
    seen: list[tuple] = []

    def spy(*args, **kwargs):
        seen.append(args)
        return orig(*args, **kwargs)

    monkeypatch.setattr(core_mod, "_resolve_distributed", spy)

    ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(bt.col("a") > 1)
    stats = ds.stats()

    assert stats.rows == 2
    assert seen, "run_profiled did not resolve the distributed routing"
    # The size-aware "auto" decision needs the plan and sources — the same three-arg call
    # `collect()`/`write()` make. Passing only ("auto",) is the bug.
    plan_source_calls = [a for a in seen if len(a) >= 3 and a[1] is not None and a[2] is not None]
    assert plan_source_calls, f"run_profiled resolved routing without a plan/sources: calls={seen}"
