"""The same operator over a bounded source and an unbounded one returns the same rows.

`.claude/rules/testing.md` names DuckDB as the oracle for *what* an operator computes.
This file pins the other half — that the answer does not depend on *how the rows arrive*.
A running fold consumes one micro-batch at a time and merges; a batch aggregate sees the
whole input at once. Nothing but a test like this one compares them, and the failure mode
when they disagree is a silently short total rather than an error.

The matrix is the cross-product `CLAUDE.md` asks for: every mergeable shape against
`{dense, nulls-and-NaN}`. It is the harness that found `files_incremental`'s required
`state_dir` — the documented two-argument call raised `TypeError` before a single row was
compared, and no executable test reached the reader because its docstring example is
`# doctest: +SKIP`.
"""

from __future__ import annotations

import datetime
import math
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

_T0 = datetime.datetime(2024, 1, 1)


def _rows(n: int, *, seed: int, sparse: bool) -> dict:
    """`n` event-time rows over five keys; `sparse` mixes in nulls, NaN and -0.0."""
    rng = random.Random(seed)
    timestamps = sorted(_T0 + datetime.timedelta(seconds=rng.randrange(120)) for _ in range(n))
    if sparse:
        values = [rng.choice([1.0, -1.0, 0.0, -0.0, 2.5, float("nan")]) for _ in range(n)]
        values = [None if rng.random() < 0.2 else v for v in values]
    else:
        values = [float(rng.randrange(-10, 10)) for _ in range(n)]
    return {
        "ts": timestamps,
        "k": [f"k{rng.randrange(5)}" for _ in range(n)],
        "v": values,
        "i": [rng.randrange(100) for _ in range(n)],
    }


def _canon(value):
    """One spelling per value: NaN compares equal to NaN, and -0.0 to 0.0."""
    if value is None:
        return "\x00"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return 0.0 if value == 0.0 else round(value, 9)
    return value


def _multiset(table: pa.Table) -> list[tuple]:
    """`table` as an order-independent, column-order-independent multiset of rows."""
    columns = [table.column(name).to_pylist() for name in sorted(table.schema.names)]
    return sorted((tuple(_canon(v) for v in row) for row in zip(*columns, strict=True)), key=repr)


# Each case is `(needs_watermark, build)`. `build` is applied to both datasets verbatim —
# the point of the test is that one pipeline expression serves both.
_CASES = {
    "group_by_sum": (False, lambda d: d.group_by("k").agg(s=bt.col("v").sum())),
    "group_by_count": (False, lambda d: d.group_by("k").agg(n=bt.col("v").count())),
    "group_by_mean": (False, lambda d: d.group_by("k").agg(m=bt.col("v").mean())),
    "group_by_min_max": (
        False,
        lambda d: d.group_by("k").agg(lo=bt.col("v").min(), hi=bt.col("v").max()),
    ),
    "group_by_stddev": (False, lambda d: d.group_by("k").agg(s=bt.col("v").std())),
    "group_by_n_unique": (False, lambda d: d.group_by("k").agg(n=bt.col("i").n_unique())),
    "group_by_sum_and_count": (
        False,
        lambda d: d.group_by("k").agg(s=bt.col("v").sum(), n=bt.col("v").count()),
    ),
    "distinct": (False, lambda d: d.select("k").distinct()),
    "filter_select": (False, lambda d: d.filter(bt.col("i") > 50).select("k", "i")),
    "window_tumbling": (
        True,
        lambda d: d.group_by(w=bt.window(bt.col("ts"), "1 minute")).agg(n=bt.col("v").count()),
    ),
    "window_keyed": (
        True,
        lambda d: d.group_by("k", w=bt.window(bt.col("ts"), "30 seconds")).agg(s=bt.col("v").sum()),
    ),
}


@pytest.mark.integration
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "nulls"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_streamed_result_equals_batch_result(tmp_path, monkeypatch, case, sparse):
    monkeypatch.setenv("BATCHER_HOME", str(tmp_path / "home"))
    needs_watermark, build = _CASES[case]
    data = _rows(200, seed=7, sparse=sparse)

    bounded = bt.from_pydict(data)
    expected = _multiset(build(bounded).collect())

    landing = tmp_path / "landing"
    landing.mkdir()
    pq.write_table(pa.table(data), str(landing / "part.parquet"))

    stream = bt.read.files_incremental(str(landing), "parquet")
    if needs_watermark:
        stream = stream.with_watermark("ts", "10 seconds")
    produced = list(build(stream).iter_batches())

    assert produced, "the streaming path emitted nothing at all"
    actual = _multiset(pa.Table.from_batches(produced, schema=produced[0].schema))
    assert actual == expected
