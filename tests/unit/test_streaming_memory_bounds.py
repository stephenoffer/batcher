"""Peak working set must not grow with the input — the property no correctness suite can see.

The streaming executor's whole claim is that a pipeline operator holds one morsel at a time,
so its peak memory is a constant independent of how many rows flow through it. Every test in
the tree checks *what* a query returns; none checked *how much memory it took to get there*,
and the gap is not theoretical:

* Before streaming became the default, `par::exec()` returned a `Vec<RecordBatch>` for every
  node, breaker or not. At sf100 TPC-H q3/q4/q5 peaked at **133 GB** and were OOM-killed.
* After it, the streaming probe was `stream.map(|m| gather_join_output_with(...))` — one input
  morsel in, one output `RecordBatch` out, however many rows that was. A 16,384-row probe
  morsel against a build side with one distinct key emitted 327 million rows as a single
  batch: **13.1 GB RSS**. The 46-test oracle suite passed throughout, because every one of
  those tests is correct and none of them fans out. As `competitive_architecture.md` puts it,
  "a correctness suite cannot see a memory property."

Both regressions would be caught here. The measurements come from the engine's own
`m_peak_bytes` (`hub.op_stats_by_kind()`) rather than from sampling RSS, so the gate is
deterministic, cheap, and attributes the growth to a named operator instead of to the process.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt  # noqa: E402  (after importorskip)
from batcher import core  # noqa: E402

#: How far the input is scaled between the small and large run. Large enough that a linear
#: term cannot hide inside measurement noise: an operator that materializes its input grows by
#: this factor, one that streams does not move at all.
_SCALE = 16

#: Peak growth a streaming operator is allowed across a `_SCALE`x input. Measured growth is
#: exactly 1.0 — the peak is one morsel either way — so this is pure headroom for morsel
#: boundary effects, and still an order of magnitude below the linear growth it guards against.
_MAX_GROWTH = 2.0


def _peaks(kinds: tuple[str, ...]) -> dict[str, int]:
    """The most recent `m_peak_bytes` for each operator kind in `kinds`."""
    by_kind = core.default_hub().op_stats_by_kind()
    return {k: by_kind[k][-1].get("m_peak_bytes", 0) for k in kinds if by_kind.get(k)}


def _run_pipeline(rows: int) -> dict[str, int]:
    """A breaker-free filter+project over `rows` rows; returns its operators' peaks."""
    table = pa.table(
        {
            "x": pa.array(range(rows), pa.int64()),
            "y": pa.array([float(i) for i in range(rows)], pa.float64()),
        }
    )
    dataset = bt.from_arrow(table.to_batches(max_chunksize=16384))
    dataset.filter(bt.col("x") % 2 == 0).select("x", z=bt.col("y") * 2.0).collect()
    return _peaks(("filter", "project"))


def _run_fanout_join(side: int) -> tuple[int, dict[str, int]]:
    """A single-key join of `side`x`side` rows, so the output is `side**2`."""
    probe = pa.table(
        {"k": pa.array([0] * side, pa.int64()), "p": pa.array(range(side), pa.int64())}
    )
    build = pa.table(
        {"k": pa.array([0] * side, pa.int64()), "b": pa.array(range(side), pa.int64())}
    )
    left = bt.from_arrow(probe.to_batches(max_chunksize=16384))
    right = bt.from_arrow(build.to_batches(max_chunksize=16384))
    out = left.join(right, on="k").collect()
    return out.num_rows, _peaks(("hash_join",))


def _assert_bounded(small: dict[str, int], large: dict[str, int], what: str) -> None:
    assert small and large, f"no {what} metrics were recorded"
    for kind, small_peak in small.items():
        large_peak = large.get(kind, 0)
        assert small_peak > 0, f"{kind} reported no working set at all"
        growth = large_peak / small_peak
        assert growth <= _MAX_GROWTH, (
            f"{kind} peak grew {growth:.1f}x ({small_peak} -> {large_peak} bytes) while the "
            f"{what} grew {_SCALE}x. A pipeline operator holds one morsel at a time, so its "
            "peak must not scale with the data — this is the shape that OOM-killed sf100."
        )


def test_a_linear_pipeline_holds_one_morsel_however_many_rows_flow_through():
    small = _run_pipeline(100_000)
    large = _run_pipeline(100_000 * _SCALE)
    _assert_bounded(small, large, "input")


def test_a_high_fanout_join_does_not_buffer_its_whole_output():
    """The 13.1 GB regression: one input morsel produced one output batch of any size.

    Fan-out is squared here — `side`x`side` rows out of `side` in — so the output grows 16x
    between the two runs while the input grows 4x. An operator that emits its result as a
    single batch grows with the output; one that morselizes does not move.
    """
    small_rows, small = _run_fanout_join(200)
    large_rows, large = _run_fanout_join(200 * 4)

    assert large_rows == small_rows * _SCALE, f"{small_rows} -> {large_rows}, expected 16x"
    _assert_bounded(small, large, "output")
