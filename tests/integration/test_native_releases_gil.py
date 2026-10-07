"""The native aggregate folds and sketch builds release the GIL while they compute.

A fleet actor runs several methods at once on its own threads, and these calls are what it
folds a partition or sketches a column with. Holding the GIL for the whole computation
stalled every other thread of the actor -- a Flight gather, a split read, a heartbeat --
serializing the concurrency the actor was granted (BT-114).

Measured directly: a Python thread spins while the native call runs on another, and the
longest gap between its iterations is compared with the call's own duration. A call that
holds the GIL freezes the spinner for the whole call; one that releases it leaves only
interpreter switch-interval gaps.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest

from batcher import col, count

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

# A call shorter than this cannot tell a GIL stall from the spinner simply being descheduled,
# which on a loaded box is a gap of up to a couple of hundred milliseconds by itself; the
# inputs below are sized so every call runs several times longer than that.
_MIN_CALL_S = 0.15


def _nat():
    from batcher._internal.native import engine

    return engine()


def _batches(n: int, groups: int) -> list[pa.RecordBatch]:
    rng = np.random.default_rng(11)
    table = pa.table(
        {
            "k": rng.integers(0, groups, n).astype("int64"),
            "v": rng.integers(0, 1000, n).astype("int64"),
        }
    )
    return table.to_batches(max_chunksize=1 << 20)


def _agg_json() -> tuple[str, str]:
    gk = json.dumps([{"expr": col("k").to_ir(), "alias": "k"}])
    ag = json.dumps([spec.to_ir(alias) for alias, spec in [("s", col("v").sum()), ("n", count())]])
    return gk, ag


def _max_stall(call: Callable[[], object]) -> tuple[float, float]:
    """`(longest gap a spinning Python thread saw, the call's own duration)`."""
    done = threading.Event()
    took: list[float] = []

    def run() -> None:
        start = time.perf_counter()
        call()
        took.append(time.perf_counter() - start)
        done.set()

    worker = threading.Thread(target=run)
    longest = 0.0
    last = time.perf_counter()
    worker.start()
    while not done.is_set():
        now = time.perf_counter()
        longest = max(longest, now - last)
        last = now
    worker.join()
    return longest, took[0]


def _cases() -> dict[str, Callable[[], object]]:
    nat = _nat()
    gk, ag = _agg_json()
    rows = _batches(4_000_000, 1_000_000)
    partials = [nat.partial_aggregate(gk, ag, [b]) for b in rows]
    return {
        "partial_aggregate": lambda: nat.partial_aggregate(gk, ag, rows),
        "combine_finalize": lambda: nat.combine_finalize(gk, ag, partials),
        "combine": lambda: nat.combine(gk, ag, partials),
        "estimate_distinct": lambda: nat.estimate_distinct("v", rows),
        "tail_quantiles": lambda: nat.tail_quantiles(["v", "k"], rows, [0.5, 0.99]),
        "tdigest_partial": lambda: nat.tdigest_partial("k", rows),
        "heavy_hitters": lambda: nat.heavy_hitters(["k", "v"], rows, 0.01),
        # The cheapest per row of these, so it gets more rows to outlast scheduler noise.
        "reservoir_sample": lambda: nat.reservoir_sample(rows * 6, 1000),
    }


@pytest.fixture(scope="module")
def cases() -> dict[str, Callable[[], object]]:
    return _cases()


@pytest.mark.parametrize(
    "name",
    [
        "partial_aggregate",
        "combine_finalize",
        "combine",
        "estimate_distinct",
        "tail_quantiles",
        "tdigest_partial",
        "heavy_hitters",
        "reservoir_sample",
    ],
)
def test_the_call_does_not_freeze_other_python_threads(cases, name: str) -> None:
    call = cases[name]
    longest, took = _max_stall(call)
    # The control: a call too quick to outlast the switch interval proves nothing either way.
    assert took >= _MIN_CALL_S, (
        f"{name} took {took * 1000:.0f} ms; the input is too small to measure"
    )
    assert longest < took / 2, (
        f"{name} froze another Python thread for {longest * 1000:.0f} ms of its "
        f"{took * 1000:.0f} ms: it holds the GIL while it computes"
    )
