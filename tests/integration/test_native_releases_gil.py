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


def _cases() -> dict[str, Callable[[int], Callable[[], object]]]:
    """Each call as a factory over `reps`, the number of times its input is repeated."""
    nat = _nat()
    gk, ag = _agg_json()
    rows = _batches(4_000_000, 1_000_000)
    partials = [nat.partial_aggregate(gk, ag, [b]) for b in rows]
    # A repeated list repeats references, not data: `rows * r` costs nothing to build.
    return {
        "partial_aggregate": lambda r: lambda: nat.partial_aggregate(gk, ag, rows * r),
        "combine_finalize": lambda r: lambda: nat.combine_finalize(gk, ag, partials * r),
        "combine": lambda r: lambda: nat.combine(gk, ag, partials * r),
        "estimate_distinct": lambda r: lambda: nat.estimate_distinct("v", rows * r),
        "tail_quantiles": lambda r: lambda: nat.tail_quantiles(["v", "k"], rows * r, [0.5, 0.99]),
        "tdigest_partial": lambda r: lambda: nat.tdigest_partial("k", rows * r),
        "heavy_hitters": lambda r: lambda: nat.heavy_hitters(["k", "v"], rows * r, 0.01),
        # The cheapest per row of these, so it gets more rows to outlast scheduler noise.
        "reservoir_sample": lambda r: lambda: nat.reservoir_sample(rows * 6 * r, 1000),
    }


def _long_enough(make: Callable[[int], Callable[[], object]]) -> Callable[[], object]:
    """The call with its input repeated until one run outlasts `_MIN_CALL_S` twice over.

    A fixed input cannot be sized for every machine: the 4M-row fold that takes several
    hundred milliseconds on a laptop took 33-81 ms on a 64-core gate node, failing the
    control below rather than measuring anything. Growing the input keeps the control's
    meaning -- the call must still run long enough to tell a stall from descheduling.
    """
    reps = 1
    while True:
        call = make(reps)
        start = time.perf_counter()
        call()
        took = time.perf_counter() - start
        if took >= 2 * _MIN_CALL_S or reps >= 64:
            return call
        reps = min(64, reps * max(2, int(2 * _MIN_CALL_S / max(took, 1e-3)) + 1))


@pytest.fixture(scope="module")
def cases() -> dict[str, Callable[[int], Callable[[], object]]]:
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
    call = _long_enough(cases[name])
    longest, took = _max_stall(call)
    # The control: a call too quick to outlast the switch interval proves nothing either way.
    assert took >= _MIN_CALL_S, (
        f"{name} took {took * 1000:.0f} ms; the input is too small to measure"
    )
    assert longest < took / 2, (
        f"{name} froze another Python thread for {longest * 1000:.0f} ms of its "
        f"{took * 1000:.0f} ms: it holds the GIL while it computes"
    )
