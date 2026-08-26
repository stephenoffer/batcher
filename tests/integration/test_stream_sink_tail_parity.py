"""Work *above* a streaming aggregate reaches a streaming sink, and matches batch.

A running fold computes the aggregate; the row-wise operators the query asked for above
it — a projection, a HAVING filter, the arithmetic an expression over aggregates lowers
to — are applied to each emitted snapshot. That is what the batch plan computes over the
whole input, because a row-wise node's output for a row depends on that row alone.

Every shape here used to be refused outright with "this plan cannot be streamed to a sink
(it has a pipeline breaker other than a top-level aggregation)". The refusal read as a
missing operator and was really a missing projection: `sum(x) / count()`, `max(v) -
min(v)`, `regr_slope`, `.select(...)` after an `agg` and a HAVING filter all lower to a
`Project`/`Filter` over the `Aggregate` rather than to one node, and the router only
recognized a bare top-level fold.

**This terminal is the one that could see it.** `tests/integration/test_stream_batch_
operator_parity.py` exercises the same shapes through `iter_batches`, which has always
materialized them — so the whole matrix stayed green while the sink refused every one. A
capability gap is invisible to a terminal that does not have it.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.float64()), ("i", pa.int64())])
_TABLE = pa.table(
    {
        "k": pa.array(["a", "a", "b", "b", "c"]),
        "v": pa.array([1.0, 2.0, 3.0, 4.0, 10.0]),
        "i": pa.array([1, 2, 3, 4, 5]),
    }
)


def _stream():
    def gen():
        yield from _TABLE.to_batches(max_chunksize=2)

    return bt.from_batches(gen, _SCHEMA, bounded=False)


def _canon(value):
    """One spelling per value: NaN compares equal to NaN, and -0.0 to 0.0.

    `regr_slope` over a single-row group is NaN on both paths, and `nan != nan` would read
    as a parity failure where the two sides agree exactly.
    """
    if value is None:
        return "\x00"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return 0.0 if value == 0.0 else round(value, 9)
    return value


def _multiset(table: pa.Table) -> list[tuple]:
    columns = [table.column(name).to_pylist() for name in sorted(table.schema.names)]
    return sorted((tuple(_canon(v) for v in row) for row in zip(*columns, strict=True)), key=repr)


# Every case is applied verbatim to a bounded and an unbounded dataset — the point being
# that one pipeline expression serves both.
_CASES = {
    "bare_aggregate": lambda d: d.group_by("k").agg(s=bt.col("v").sum()),
    "agg_then_select": lambda d: d.group_by("k").agg(s=bt.col("v").sum()).select("s"),
    "agg_then_with_columns": (
        lambda d: d.group_by("k").agg(s=bt.col("v").sum()).with_columns(t=bt.col("s") * 2)
    ),
    "having": lambda d: d.group_by("k").agg(s=bt.col("v").sum()).filter(bt.col("s") > 4),
    "expr_over_aggregates": lambda d: d.group_by("k").agg(a=bt.col("v").sum() / bt.count()),
    "expr_spread": lambda d: d.group_by("k").agg(a=bt.col("v").max() - bt.col("v").min()),
    "regr_slope": lambda d: d.group_by("k").agg(a=bt.regr_slope(bt.col("v"), bt.col("i"))),
    "distinct_then_select": lambda d: d.select("k").distinct().select("k"),
    "chained_tail": (
        lambda d: d.group_by("k").agg(s=bt.col("v").sum()).filter(bt.col("s") > 0).select("s")
    ),
}


@pytest.mark.integration
@pytest.mark.parametrize("case", sorted(_CASES))
def test_complete_mode_sink_equals_batch(case):
    """`complete` re-emits the whole snapshot, so the sink holds exactly the batch answer."""
    build = _CASES[case]
    expected = _multiset(build(bt.from_arrow(_TABLE)).collect())

    sink = f"tail_complete_{case}"
    query = build(_stream()).write.memory(
        sink, trigger=bt.Trigger.available_now(), output_mode="complete"
    )
    query.await_termination()

    assert _multiset(bt.read_memory(sink).collect()) == expected


@pytest.mark.integration
def test_a_grouped_update_stream_converges_on_the_batch_answer():
    """`update` emits only changed rows, so each group's final value is its batch value.

    Deliberately not the `complete` assertion: an append-only memory sink accumulates every
    revision, which is what the mode means. The invariant that holds is per key — the last
    value emitted for a group equals the batch aggregate for that group.
    """
    build = _CASES["expr_over_aggregates"]
    batch_result = build(bt.from_arrow(_TABLE)).collect().to_pydict()
    expected = dict(zip(batch_result["k"], batch_result["a"], strict=True))

    query = build(_stream()).write.memory(
        "tail_update", trigger=bt.Trigger.available_now(), output_mode="update"
    )
    query.await_termination()

    emitted = bt.read_memory("tail_update").collect().to_pydict()
    last: dict[str, float] = {}
    for key, value in zip(emitted["k"], emitted["a"], strict=True):
        last[key] = value
    assert last == expected


@pytest.mark.integration
def test_the_tail_applies_to_the_identity_row_an_empty_stream_owes():
    """A keyless aggregate over an empty stream emits its identity row — projected.

    The identity row is a fold result like any other. Emitting it *unprojected* would put
    the pre-projection columns into the sink and disagree with `collect()` on exactly the
    shape that fallback exists to keep in agreement.
    """
    empty = bt.from_batches(lambda: iter(()), _SCHEMA, bounded=False)
    expected = _multiset(bt.from_arrow(_TABLE.slice(0, 0)).agg(n=bt.count()).select("n").collect())

    query = (
        empty.agg(n=bt.count())
        .select("n")
        .write.memory("tail_empty", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    query.await_termination()

    got = bt.read_memory("tail_empty").collect()
    assert got.column_names == ["n"]
    assert _multiset(got) == expected


@pytest.mark.integration
def test_append_on_a_windowed_aggregate_refuses_a_tail_rather_than_projecting_a_partial():
    """A closed window is emitted once and never revised, so a tail has nothing sound to do.

    Refused with a message naming the alternative, rather than silently applying the tail
    to a partial result.
    """
    import datetime

    from batcher._internal.errors import PlanError

    schema = pa.schema([("ts", pa.timestamp("us")), ("v", pa.float64())])
    table = pa.table(
        {
            "ts": pa.array([datetime.datetime(2024, 1, 1)], type=pa.timestamp("us")),
            "v": pa.array([1.0]),
        }
    )

    stream = bt.from_batches(lambda: iter(table.to_batches()), schema, bounded=False)
    windowed = (
        stream.with_watermark("ts", "10 seconds")
        .group_by(w=bt.window(bt.col("ts"), "1 minute"))
        .agg(n=bt.col("v").count())
        .select("n")
    )
    with pytest.raises(PlanError, match="cannot carry work above the aggregate"):
        windowed.write.memory(
            "tail_append", trigger=bt.Trigger.available_now(), output_mode="append"
        )


# --- the mapped aggregate, which peels the same way ----------------------


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    import pyarrow.compute as pc

    return pa.record_batch(
        {"k": batch.column("k"), "v": pc.multiply(batch.column("v"), 2), "i": batch.column("i")}
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "case", ["bare_aggregate", "agg_then_select", "having", "expr_over_aggregates"]
)
def test_a_map_batches_aggregate_streams_under_a_tail(case):
    """`map_batches → agg → select` is the ML streaming shape: inference, then a rollup.

    The launcher recognizes a mapped aggregate so the UDF runs in Python and the fold
    consumes what it returns. That test read the *top* node, so a `select` or HAVING above
    the aggregate made the shape unrecognizable — and the result was worse than the refusal
    it replaced: with no per-batch runner built, the fold tried to lower the `MapBatches`
    and raised ``NotImplementedError: map_batches is executed in Python, not lowered to the
    engine IR``, an internal message about the wire contract.
    """
    build = _CASES[case]
    mapped = bt.from_arrow(_TABLE).map_batches(_double)
    expected = _multiset(build(mapped).collect())

    sink = f"tail_mapped_{case}"
    query = build(_stream().map_batches(_double)).write.memory(
        sink, trigger=bt.Trigger.available_now(), output_mode="complete"
    )
    query.await_termination()

    assert _multiset(bt.read_memory(sink).collect()) == expected


# --- the other terminal, for contrast --------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("case", sorted(_CASES))
def test_iter_batches_agrees_with_the_sink_on_every_tail_shape(case):
    """The same shapes through `iter_batches`, which has always materialized them.

    Kept beside the sink assertions rather than trusted separately: this terminal is why
    the gap survived so long — it answered every one of these correctly while the sink
    refused them, so a stream-vs-batch matrix driven by `iter_batches` stayed green. The
    value of the pair is that a future change has to move both together.
    """
    build = _CASES[case]
    expected = _multiset(build(bt.from_arrow(_TABLE)).collect())

    produced = list(build(_stream()).iter_batches())
    assert produced, "the streaming path emitted nothing at all"
    actual = _multiset(pa.Table.from_batches(produced, schema=produced[0].schema))
    assert actual == expected
