"""The fast path closes the learning loop -- on the first run *and* on the replay.

`fast_path` skips Carbonite admission, adaptive sizing, the event bus and profile assembly.
It used to skip the whole write side of the learned-stats loop with them, which made the
cross-query moat the price of not paying admission on a query too small to need it. The two
are independent: admission is skipped because `eligible` bounds the input below any envelope
it could have defended, and that argument says nothing about whether the run is *measured*.

These pin the half that is easy to reopen by accident. The replay path (`prepared.Prepared`)
is the one that matters most -- a hot shape spends its life there, and a loop that records
the first execution of a shape and nothing after it is the half-loop the module docstring
argues is worse than none.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.orchestration import prepared
from batcher.config import active_config, set_config


@pytest.fixture
def fast_path_on():
    cfg = active_config()
    set_config(cfg.replace(execution=dataclasses.replace(cfg.execution, fast_path=True)))
    prepared.clear()
    try:
        yield
    finally:
        set_config(cfg)
        prepared.clear()


def _table():
    return pa.table({"k": [1, 2, 1, 2, 3], "v": [1.0, 2.0, 3.0, 4.0, 5.0]})


def _loop_counters(hub):
    """The hub's two monotonic write counters -- the operator history and the learned params.

    `version` bumps on every recorded feedback row (the per-operator `ExecMetrics` that
    calibrate the cost model); `params_version` bumps on every learned-parameter write (the
    measured cardinality and selectivity). Both must move, because the two halves are
    written by different call sites and either can be reopened alone.
    """
    return hub.version, hub.params_version


def test_the_fast_path_is_actually_taken(fast_path_on):
    """Precondition: without this the two tests below prove nothing about the fast path."""
    from batcher.api.orchestration.fast_path import eligible

    ds = bt.from_arrow(_table())
    assert eligible(
        ds.filter(bt.col("v") > 1.0)._plan,
        ds._sources,
        distributed=False,
        adaptive=False,
        spill=False,
        backend="cpu",
        cache=None,
    )


def test_a_replayed_query_is_still_measured(fast_path_on):
    """The second run of a shape is a prepared-cache hit; it must still feed the hub."""
    from batcher import core

    hub = core.default_hub()
    ds = bt.from_arrow(_table())
    query = lambda: ds.filter(bt.col("v") > 1.0).collect()  # noqa: E731

    query()  # first: derives and remembers
    before = _loop_counters(hub)
    query()  # second: served from the prepared cache
    after = _loop_counters(hub)
    assert after[0] > before[0], (
        "a prepared-cache hit recorded no operator metrics -- the cost model stops "
        "calibrating on exactly the shapes that run most"
    )
    assert after[1] > before[1], (
        "a prepared-cache hit recorded no learned parameters -- the measured cardinality "
        "loop is closed on the first run of a shape and open on every one after it"
    )


def test_the_result_is_unchanged_by_recording(fast_path_on):
    """Recording steers later plans only; this run's rows, names and types are untouched."""
    ds = bt.from_arrow(_table())
    got = ds.filter(bt.col("v") > 1.0).collect()
    cfg = active_config()
    set_config(cfg.replace(execution=dataclasses.replace(cfg.execution, fast_path=False)))
    try:
        expected = bt.from_arrow(_table()).filter(bt.col("v") > 1.0).collect()
    finally:
        set_config(cfg)
    assert got.to_pydict() == expected.to_pydict()
    assert got.schema == expected.schema
