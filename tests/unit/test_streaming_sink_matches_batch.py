"""A streaming query's answer must be the batch answer, or the query must be refused.

Batcher has two consumers of a streaming plan and they took different routes through it:
`iter_batches` picks a strategy in `api/terminal/stream/dispatch.py`, and `ds.write(...)`
to a sink picks a `MicroBatchProcessor` in `core/streaming_query/processors.py`. Both
answer "can this shape stream?" and they answered differently.

The router required a **breaker-free input** beneath a top-level aggregate; the sink path
tested `isinstance(plan, (Aggregate, Distinct))` and looked no further. So
``ds.distinct().agg(n=count())`` streamed to a sink built a running fold whose per-batch
work re-ran the `Distinct` *inside each micro-batch* — deduplicating each batch on its own
and counting the survivors. Over ``[1,1]`` then ``[1,2]`` it returned **3** where the same
query collected returns **2**. Nothing was red: `iter_batches` took the materializing path
for that plan and got the right answer, so no test compared the two.

Every case below is therefore a *trichotomy*, not an equality: the sink path must either
produce the batch answer or refuse with `PlanError`. What it must never do is produce a
third thing. The refusals are asserted as refusals rather than skipped, because a shape
silently becoming unsupported is its own regression.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.core.streaming_query import make_processor

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("k", pa.int64()), ("v", pa.int64())])

#: Deliberately split so a value repeats *across* batches as well as within one. A
#: per-batch fold that is wrong only across boundaries is invisible on a single batch.
_BATCHES = [
    [{"k": 1, "v": 10}, {"k": 1, "v": 10}, {"k": 2, "v": 20}],
    [{"k": 1, "v": 10}, {"k": 2, "v": 21}, {"k": 3, "v": 30}],
]


def _batches() -> list[pa.RecordBatch]:
    return [pa.RecordBatch.from_pylist(rows, schema=_SCHEMA) for rows in _BATCHES]


def _stream():
    """The same rows as an unbounded source — finite, so a test can drain it."""
    return bt.from_batches(_batches, _SCHEMA, bounded=False)


def _bounded():
    """The same rows as a bounded relation — the oracle every case is held against."""
    return bt.from_arrow(pa.Table.from_batches(_batches()))


def _sorted_pydict(table: pa.Table) -> dict:
    """Column dict with rows in a canonical order, so comparison is order-independent."""
    if table.num_rows == 0:
        return {name: [] for name in table.schema.names}
    order = sorted(range(table.num_rows), key=lambda i: repr(table.slice(i, 1).to_pylist()))
    return table.take(order).to_pydict()


def _drive_to_sink(plan, output_mode: str = "complete") -> dict:
    """Fold every micro-batch through the sink path's processor; return the final emission.

    This is what `StreamingQueryEngine` does per trigger, minus the sink — the processor is
    the whole of the query's semantics, so driving it directly is the sharpest place to
    compare against batch.
    """
    processor = make_processor(plan, output_mode, None, None)
    emitted: list[pa.RecordBatch] = []
    for batch in _batches():
        emitted.extend(processor.process(batch))
    finalize = getattr(processor, "finalize", None)
    if finalize is not None:
        emitted.extend(finalize())
    if not emitted:
        return {}
    # Complete mode re-emits the whole running result each micro-batch, so the answer is
    # the last emission, not the concatenation of all of them.
    return _sorted_pydict(pa.Table.from_batches([emitted[-1]]))


# --- the shapes that must agree ----------------------------------------------------


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("grouped_agg", lambda ds: ds.group_by("k").agg(total=bt.col("v").sum())),
        ("global_agg", lambda ds: ds.agg(total=bt.col("v").sum())),
        ("distinct", lambda ds: ds.distinct()),
        ("distinct_after_project", lambda ds: ds.select("k").distinct()),
        ("agg_over_filter", lambda ds: ds.filter(bt.col("v") > 10).group_by("k").agg(n=bt.count())),
        ("multi_agg", lambda ds: ds.group_by("k").agg(s=bt.col("v").sum(), n=bt.count())),
    ],
)
def test_the_sink_path_returns_the_batch_answer(label, build):
    """The shapes the mergeable fold genuinely covers must equal `collect()` exactly."""
    expected = _sorted_pydict(build(_bounded()).collect())
    assert _drive_to_sink(build(_stream())._plan) == expected, label


@pytest.mark.parametrize(
    ("label", "build"),
    [
        # The bug: a `Distinct` beneath the aggregate is re-run per micro-batch, so the
        # count is of per-batch survivors rather than of the stream's distinct rows.
        ("agg_over_distinct", lambda ds: ds.distinct().agg(n=bt.count())),
        ("agg_over_distinct_grouped", lambda ds: ds.distinct().group_by("k").agg(n=bt.count())),
        # A sort beneath is order-only and would happen to give the right aggregate, but
        # the fold cannot prove that in general, so it is refused with the rest.
        ("agg_over_sort", lambda ds: ds.sort("v").group_by("k").agg(n=bt.count())),
        ("agg_over_limit", lambda ds: ds.limit(4).group_by("k").agg(n=bt.count())),
    ],
)
def test_a_breaker_beneath_the_fold_is_refused_not_miscomputed(label, build):
    """A second breaker under the aggregate is refused, because the fold would re-run it.

    Each of these used to build an `AggregateProcessor` and return an answer that is not
    the batch answer. Refusing is the correct outcome: the running fold's per-batch plan is
    `agg.input`, so any breaker in there is applied to a micro-batch rather than the stream.
    """
    with pytest.raises(PlanError, match="breaker-free input"):
        _drive_to_sink(build(_stream())._plan)


def test_the_refusal_names_the_operator_it_found():
    """An error that says "a pipeline breaker" and not which one sends the reader hunting."""
    with pytest.raises(PlanError, match="distinct"):
        _drive_to_sink(_stream().distinct().agg(n=bt.count())._plan)


def test_a_keyed_distinct_keeps_its_own_diagnosis():
    """`DISTINCT ON` has a specific way out (a watermark dedup) and must keep pointing at it,
    rather than being swept into the generic breaker refusal.

    Built over a *bounded* source on purpose. `Dataset.distinct(subset=...)` refuses an
    unbounded input at build time, so the only way to reach `make_processor` with this shape
    is a bounded relation drained as a stream (`trigger=available_now`) — which is a real
    call, and the only one that exercises this branch.
    """
    plan = _bounded().distinct(subset=["k"])._plan
    with pytest.raises(PlanError, match="drop_duplicates_within_watermark"):
        make_processor(plan, "complete", None, None)


def test_update_mode_still_folds_the_shapes_it_always_did():
    """The gate must not have narrowed the output modes, only the plan shapes."""
    plan = _stream().group_by("k").agg(total=bt.col("v").sum())._plan
    assert make_processor(plan, "update", None, None) is not None


# --- distinct().limit(n) streams, and stops -----------------------------------------


@pytest.mark.parametrize(("n", "offset"), [(1, 0), (2, 0), (3, 0), (2, 1), (10, 0)])
def test_distinct_limit_streams_the_same_rows_batch_returns(n, offset):
    """`distinct().limit(n)` is bounded state and a terminating read, and used to be refused.

    The router's `Limit` branch needs a breaker-free input and a `Distinct` is a breaker, so
    the pair fell through to "this plan must materialize" — for the most ordinary way anyone
    inspects an unfamiliar topic. The rows must match batch exactly, offset included, because
    the capped operator's contract is the *first* n distinct rows in input order rather than
    an arbitrary n of them.
    """
    streamed = pa.Table.from_batches(
        list(_stream().distinct().limit(n, offset=offset).iter_batches())
    ).to_pylist()
    assert streamed == _bounded().distinct().limit(n, offset=offset).collect().to_pylist()


def test_distinct_limit_stops_reading_an_endless_source():
    """The bound is only real if the read stops; on an unbounded source that is the query
    terminating at all. Asserted as batches consumed, not as elapsed time."""
    import itertools

    schema = pa.schema([("k", pa.int64())])
    reads = {"n": 0}

    def endless():
        for i in itertools.count():
            reads["n"] += 1
            yield pa.RecordBatch.from_pylist([{"k": i % 5}], schema=schema)

    ds = bt.from_batches(endless, schema, bounded=False)
    out = pa.Table.from_batches(list(ds.distinct().limit(3).iter_batches())).to_pylist()

    assert out == [{"k": 0}, {"k": 1}, {"k": 2}]
    # Three distinct values need three one-row batches; a fourth read would mean the exit
    # fired late, and no exit at all would mean this test never returned.
    assert reads["n"] == 3, reads


def test_distinct_head_is_a_bounded_peek_like_filter_head():
    """The materializing gate and the router must agree about what is finite.

    `head(n)` and `count()` do not go through the `iter_batches` router — they have their own
    finiteness gate — so teaching the router to stream `distinct().limit(n)` left the two
    disagreeing: `filter(...).head(20)` returned rows off a live topic and
    `distinct().head(20)` beside it raised "this operation materializes the full result",
    though both terminate and both hold n rows. The caller had no way to predict which.
    """
    from batcher.api.terminal.core import _is_bounded_peek

    assert _is_bounded_peek(_stream().distinct().limit(3)._plan)
    assert _is_bounded_peek(_stream().filter(bt.col("v") > 0).limit(3)._plan)
    # Still finite-but-unreachable: top-N is not known until the last row has arrived.
    assert not _is_bounded_peek(_stream().sort("v").limit(3)._plan)


def test_distinct_head_over_a_stream_returns_what_batch_returns():
    """The peek must be the batch answer's prefix, not merely n rows of something."""
    streamed = _stream().distinct().head(2).to_pydict()
    assert streamed == _bounded().distinct().head(2).to_pydict()


def test_a_keyed_distinct_head_is_not_a_bounded_peek():
    """A `DISTINCT ON` survivor can be replaced by a later row, so no prefix settles it."""
    from batcher.api.terminal.core import _is_bounded_peek
    from batcher.plan.logical import Distinct, Limit

    keyed = Limit(Distinct(_stream()._plan, keys=("k",)), 3)
    assert not _is_bounded_peek(keyed)


def test_a_fused_distinct_limit_takes_the_capped_driver_not_the_running_fold():
    """The shape `fuse_limit_into_distinct` produces must route to the capped driver.

    The uncapped fold's `as_aggregate` raises on a fused limit, so if the general
    `(Aggregate, Distinct)` branch were tested first this would be an error rather than a
    result — which is why branch order here is a contract and not a style choice.
    """
    from batcher.plan.logical import Distinct

    capped = Distinct(_stream().select("k")._plan, limit=2)
    ds = _stream()
    from batcher.api.terminal.stream import _iter_batches

    out = pa.Table.from_batches(
        list(_iter_batches(capped, list(ds._sources), capped.available_columns()))
    ).to_pylist()
    assert out == [{"k": 1}, {"k": 2}]


# --- the refusal must name the operator that actually blocks -------------------------


@pytest.mark.parametrize(
    ("build", "culprit", "innocent"),
    [
        (lambda ds: ds.sort("v").group_by("k").agg(n=bt.count()), "sort", "aggregate"),
        (lambda ds: ds.distinct().group_by("k").agg(n=bt.count()), "distinct", "aggregate"),
        (lambda ds: ds.sort("v").filter(bt.col("v") > 0), "sort", "filter"),
    ],
    ids=["agg_over_sort", "agg_over_distinct", "filter_over_sort"],
)
def test_the_router_refusal_names_the_blocking_operator_not_the_root(build, culprit, innocent):
    """The refusal named the plan's *root*, which is routinely not the thing that blocks.

    ``ds.sort("t").group_by("a").agg(...)`` reported "its top-level Aggregate forces the plan
    to materialize" — and a streaming aggregate is exactly the shape that does stream. The
    reader was pointed at the one operator in their query that was fine while the `sort`
    beneath it went unmentioned, and the suggested fixes listed shapes their plan already had.
    """
    from batcher.api.terminal.stream.dispatch import _unstreamable_reason

    reason = _unstreamable_reason(build(_stream())._plan)
    # Only the *diagnosis* half is under test. The half after "Restructure" lists the shapes
    # that do stream, and naming `filter` or `aggregate` there is the advice, not the blame.
    diagnosis = reason.split("Restructure")[0]
    assert culprit in diagnosis, reason
    assert innocent not in diagnosis, (
        f"the refusal blames {innocent!r}, which is not what stops this plan streaming"
    )


def test_the_refusal_does_not_repeat_a_repeated_operator():
    """Three sorts must read as "sort", not "sort and sort and sort"."""
    from batcher.api.terminal.stream.dispatch import _unstreamable_reason

    plan = _stream().sort("v").sort("k").sort("v")._plan
    assert _unstreamable_reason(plan).count("sort") == 1


# --- the two routers must agree on which single-source streams exist ----------------

#: Every single-source shape either router has an opinion about, each built twice — once
#: over a stream and once over the same rows bounded — so the pair below can be compared.
_SHAPES = [
    ("scan", lambda ds: ds),
    ("filter", lambda ds: ds.filter(bt.col("v") > 10)),
    ("project", lambda ds: ds.select("k")),
    ("with_columns", lambda ds: ds.with_columns(d=bt.col("v") * 2)),
    ("filter_project", lambda ds: ds.filter(bt.col("v") > 10).select("k")),
    ("grouped_agg", lambda ds: ds.group_by("k").agg(total=bt.col("v").sum())),
    ("global_agg", lambda ds: ds.agg(total=bt.col("v").sum())),
    ("distinct", lambda ds: ds.distinct()),
    ("agg_over_distinct", lambda ds: ds.distinct().agg(n=bt.count())),
    ("agg_over_sort", lambda ds: ds.sort("v").group_by("k").agg(n=bt.count())),
    ("sort", lambda ds: ds.sort("v")),
    ("topn", lambda ds: ds.sort("v").limit(2)),
    ("limit", lambda ds: ds.limit(3)),
    ("distinct_limit", lambda ds: ds.distinct().limit(2)),
    ("union_distinct", lambda ds: ds.union(ds, distinct=True)),
    ("sample_n", lambda ds: ds.sample(n=2, seed=1)),
    ("sample_fraction", lambda ds: ds.sample(fraction=0.5, seed=1)),
    ("unnest_free_project", lambda ds: ds.select("k", "v").filter(bt.col("k") > 0)),
]


def _router_streams(plan, sources) -> bool:
    """Whether `iter_batches`'s router will stream this plan rather than refuse it."""
    from batcher.api.terminal.stream import _iter_batches

    try:
        for _ in _iter_batches(plan, list(sources), plan.available_columns()):
            pass
    except PlanError:
        return False
    return True


def _sink_streams(plan) -> bool:
    """Whether `ds.write(...)` can stream this plan, by either of the launcher's two routes.

    The launcher takes a plan down one of two paths: `_is_driver_shape` sends the shapes whose
    state is retained *rows* (a limit, a session window, a watermark dedup) through the
    `iter_batches` router itself, and everything else through a `MicroBatchProcessor`. A
    helper that modelled only the second would report the first as unsupported, which is a
    property of the helper rather than of the engine.

    `run_batch` is supplied because the conductor always supplies it — a stateless plan's
    per-batch runner is built by `_build_run_batch` before `make_processor` is called.
    """
    from batcher.api.streaming._launch import _is_driver_shape

    if _is_driver_shape(plan):
        return True
    for mode in ("append", "complete", "update"):
        try:
            make_processor(plan, mode, lambda batch: [batch], None)
        except PlanError:
            continue
        return True
    return False


#: Shapes the two routers disagree on *by design*, with the reason. Anything not listed
#: here must agree, and anything listed must actually still disagree — a stale exemption
#: is how a capability quietly regresses.
_INTENTIONAL_DISAGREEMENTS: dict[str, str] = {}


@pytest.mark.parametrize(("label", "build"), _SHAPES, ids=[s[0] for s in _SHAPES])
def test_the_two_streaming_routers_agree_on_what_streams(label, build):
    """`iter_batches` and `ds.write` must not disagree about which streams exist.

    They are two consumers of one plan, and every disagreement between them is either a
    capability a user can reach one way and not the other, or — as with `agg_over_distinct`
    — a shape one of them runs with different semantics. Neither is visible from inside
    either path, which is why the contract is asserted across them rather than within one.
    """
    ds = build(_stream())
    router = _router_streams(ds._plan, ds._sources)
    sink = _sink_streams(ds._plan)
    if label in _INTENTIONAL_DISAGREEMENTS:
        assert router != sink, (
            f"{label} is listed as an intentional disagreement "
            f"({_INTENTIONAL_DISAGREEMENTS[label]}) but the two routers now agree — "
            "drop the exemption"
        )
        return
    assert router == sink, (
        f"{label}: iter_batches streams={router}, the sink path streams={sink}. One of the "
        "two grew or lost a shape the other did not. If the difference is deliberate, add it "
        "to _INTENTIONAL_DISAGREEMENTS with the reason."
    )
