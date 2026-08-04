"""Enriching a stream from a table that does not move.

The most common thing anyone does to a stream, and Batcher refused it outright: a `Join`
is a pipeline breaker, so the router saw an unbounded input beneath a breaker and raised
"the plan must materialize". The cookbook's advice was to hand-roll the lookup inside
`map_batches` — which means writing the join yourself, holding the dimension yourself, and
getting the null semantics of an outer join right yourself.

These tests pin the two things that make it sound rather than merely convenient: that the
per-batch join equals the join over the whole stream (because an equi-join is per-row on
the stream side), and that the join types preserving the *static* side are refused rather
than silently wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.int64())])


def _stream(rows: list[list[tuple[str, int]]] | None = None):
    """An unbounded source delivering `rows` one micro-batch at a time."""
    batches = rows if rows is not None else [[("a", 1), ("b", 2)], [("c", 3), ("a", 4)]]

    def feed():
        for batch in batches:
            yield pa.record_batch(
                {"k": [k for k, _ in batch], "v": [v for _, v in batch]}, schema=_SCHEMA
            )

    return bt.from_batches(feed, _SCHEMA, bounded=False)


def _dim():
    return bt.from_pydict({"k": ["a", "b"], "label": ["A", "B"]})


def _rows(dataset) -> list[dict]:
    out: list[dict] = []
    for batch in dataset.iter_batches():
        out.extend(batch.to_pylist())
    return out


def _key(row: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in row.items()))


@pytest.mark.integration
def test_an_inner_join_enriches_every_matching_row():
    rows = _rows(_stream().join(_dim(), on="k", how="inner"))
    assert sorted((r["k"], r["v"], r["label"]) for r in rows) == [
        ("a", 1, "A"),
        ("a", 4, "A"),
        ("b", 2, "B"),
    ]


@pytest.mark.integration
def test_a_left_outer_keeps_the_unmatched_stream_row():
    """The whole reason left-outer is safe here: the static side is already complete, so a
    stream row is known unmatched the instant it arrives."""
    rows = _rows(_stream().join(_dim(), on="k", how="left"))
    assert sorted((r["k"], r["v"], r["label"]) for r in rows) == [
        ("a", 1, "A"),
        ("a", 4, "A"),
        ("b", 2, "B"),
        ("c", 3, None),
    ]


@pytest.mark.integration
def test_a_right_outer_works_with_the_stream_on_the_right():
    """The mirror image, and the one that would be wrong the other way round."""
    rows = _rows(_dim().join(_stream(), on="k", how="right"))
    assert sorted((r["k"], r["v"], r["label"]) for r in rows) == [
        ("a", 1, "A"),
        ("a", 4, "A"),
        ("b", 2, "B"),
        ("c", 3, None),
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("how", "expected"), [("semi", [("a", 1), ("a", 4), ("b", 2)]), ("anti", [("c", 3)])]
)
def test_semi_and_anti_filter_the_stream_by_the_dimension(how, expected):
    rows = _rows(_stream().join(_dim(), on="k", how=how))
    assert sorted((r["k"], r["v"]) for r in rows) == expected


@pytest.mark.integration
@pytest.mark.parametrize("how", ["full", "outer"])
def test_a_full_outer_is_refused_because_it_preserves_the_static_side(how):
    with pytest.raises(PlanError, match="preserves the right"):
        _rows(_stream().join(_dim(), on="k", how=how))


@pytest.mark.integration
def test_a_left_outer_with_the_stream_on_the_right_is_refused():
    """It preserves the static side, which would mean holding every dimension row until the
    stream ended to learn which never matched."""
    with pytest.raises(PlanError, match="preserves the left"):
        _rows(_dim().join(_stream(), on="k", how="left"))


@pytest.mark.integration
def test_a_right_outer_with_the_stream_on_the_left_is_refused():
    with pytest.raises(PlanError, match="preserves the right"):
        _rows(_stream().join(_dim(), on="k", how="right"))


@pytest.mark.integration
def test_the_refusal_names_the_combinations_that_do_work():
    """A refusal that does not say what to do instead is a dead end."""
    with pytest.raises(PlanError, match="stream on the left"):
        _rows(_stream().join(_dim(), on="k", how="full"))


@pytest.mark.integration
def test_an_empty_dimension_still_produces_the_right_schema():
    """A fresh deployment, or a filter that matched nothing. An inner join yields nothing;
    a left outer must still yield the stream rows with typed nulls."""
    empty = bt.from_pydict({"k": ["z"], "label": ["Z"]}).filter(bt.col("k") == "nothing")
    rows = _rows(_stream().join(empty, on="k", how="left"))
    assert sorted((r["k"], r["v"], r["label"]) for r in rows) == [
        ("a", 1, None),
        ("a", 4, None),
        ("b", 2, None),
        ("c", 3, None),
    ]


@pytest.mark.integration
def test_the_static_side_may_carry_its_own_pipeline():
    """It is read through the ordinary router, so its filters and projections run exactly
    as they would anywhere else — including its own breakers."""
    dim = (
        bt.from_pydict({"k": ["a", "a", "b"], "n": [1, 2, 10]})
        .group_by("k")
        .agg(total=bt.col("n").sum())
    )
    rows = _rows(_stream().join(dim, on="k", how="inner"))
    assert sorted((r["k"], r["v"], r["total"]) for r in rows) == [
        ("a", 1, 3),
        ("a", 4, 3),
        ("b", 2, 10),
    ]


@pytest.mark.integration
def test_per_batch_joining_equals_joining_the_whole_stream():
    """The soundness argument, tested rather than asserted: the stream's batch boundaries
    must not be observable in the result."""
    rows = [[("a", 1), ("b", 2)], [("c", 3), ("a", 4)]]
    flat = [pair for batch in rows for pair in batch]
    streamed = _rows(_stream(rows).join(_dim(), on="k", how="left"))
    bounded = _rows(
        bt.from_pydict({"k": [k for k, _ in flat], "v": [v for _, v in flat]}).join(
            _dim(), on="k", how="left"
        )
    )
    assert sorted(map(_key, streamed)) == sorted(map(_key, bounded))


@pytest.mark.integration
def test_a_one_row_batching_of_the_same_stream_gives_the_same_answer():
    """Batch size is a scheduling choice, so it must not move a single row."""
    coarse = _rows(_stream([[("a", 1), ("b", 2), ("c", 3)]]).join(_dim(), on="k", how="left"))
    fine = _rows(_stream([[("a", 1)], [("b", 2)], [("c", 3)]]).join(_dim(), on="k", how="left"))
    assert sorted(map(_key, coarse)) == sorted(map(_key, fine))


@pytest.mark.integration
def test_an_enriched_stream_writes_to_a_sink():
    """The point of the operator: the enrichment is the pipeline, not a debugging aid."""
    query = (
        _stream()
        .join(_dim(), on="k", how="inner")
        .write.memory("static_join_sink", trigger=bt.Trigger.available_now())
    )
    assert query.await_termination(timeout=60) is True
    written = bt.read_memory("static_join_sink").to_pydict()
    assert sorted(zip(written["k"], written["v"], written["label"], strict=True)) == [
        ("a", 1, "A"),
        ("a", 4, "A"),
        ("b", 2, "B"),
    ]


@pytest.mark.integration
def test_two_streams_still_route_to_the_interval_join_not_this():
    """`stream_static_sides` must not claim a join where neither side is static — that is
    `join_stream`'s watermark-bounded operator, with entirely different semantics."""
    from batcher.api.terminal.stream.static_join import stream_static_sides

    left, right = _stream(), _stream()
    plan = left.join(right, on="k", how="inner")._plan
    assert stream_static_sides(plan, left._sources + right._sources) is None


@pytest.mark.integration
def test_distributed_true_yields_the_same_rows():
    """It is driver-executed, so asking for the cluster changes nothing about the answer.
    Pinned so that teaching it to fan out later has to keep producing this."""
    single = _rows(_stream().join(_dim(), on="k", how="left"))
    fanned = []
    for batch in _stream().join(_dim(), on="k", how="left").iter_batches(distributed=True):
        fanned.extend(batch.to_pylist())
    assert sorted(map(_key, single)) == sorted(map(_key, fanned))
