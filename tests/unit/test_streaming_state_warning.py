"""A streaming query says what it will do to memory at `start()`, not at the OOM.

Kyber could already name the operators whose state nothing releases
(`kyber.streaming.retains_unbounded_state`), and until this warning existed the analysis
had no caller at all. The engine's only defence was the runtime cap, which fires when the
retained state has already reached the memory envelope — a real backstop that reports a
query which has been running for hours as a resource error, at the moment there is nothing
left to do about it.

The same fact is a property of the plan, knowable before the first row. These tests pin
both halves of that: the shapes that leak are named with their remedy, and the shapes that
are bounded stay silent. The second half is the one that decides whether anyone reads the
warning, because a warning that fires on `head(10)` is one people learn to filter out.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PerformanceWarning
from batcher.api.streaming._diagnostics import warn_if_state_is_unbounded

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("k", pa.int64()), ("v", pa.int64())])
_BASE = dt.datetime(2024, 1, 1)


def _batches():
    return [pa.RecordBatch.from_pylist([{"ts": _BASE, "k": 1, "v": 2}], schema=_SCHEMA)]


def _stream():
    return bt.from_batches(lambda: iter(_batches()), _SCHEMA, bounded=False)


def _bounded():
    return bt.from_arrow(pa.Table.from_batches(_batches()))


def _warnings_for(ds) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_state_is_unbounded(ds._plan, list(ds._sources))
    return [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning)]


# --- the shapes that leak, and the remedy each is given ------------------------------


@pytest.mark.parametrize(
    ("label", "build", "remedy"),
    [
        ("grouped_agg", lambda d: d.group_by("k").agg(n=bt.count()), "with_watermark"),
        ("distinct", lambda d: d.distinct(), "drop_duplicates_within_watermark"),
        ("full_sort", lambda d: d.sort("k"), "top-N"),
        (
            "window",
            lambda d: d.with_columns(r=col("v").sum().over(partition_by="k")),
            "never closes a partition",
        ),
        ("distinct_union", lambda d: d.union(d, distinct=True), "UNION ALL"),
    ],
)
def test_a_leaking_operator_is_named_with_the_bound_that_would_fix_it(label, build, remedy):
    """ "This leaks" without "and here is the operator that does not" leaves the reader
    nothing to do but stop using streaming."""
    messages = _warnings_for(build(_stream()))
    assert messages, f"{label} retains state nothing releases and went unreported"
    assert remedy in messages[0], messages[0]


def test_the_warning_says_what_happens_if_it_is_ignored():
    """The consequence is what makes it actionable rather than noise."""
    (message,) = _warnings_for(_stream().group_by("k").agg(n=bt.count()))
    assert "streaming_state_max_bytes" in message
    assert "bounded test" in message, "the reason it is invisible locally is the whole point"


def test_several_leaking_nodes_of_one_kind_are_one_story():
    """A plan with two dedups has one remedy, not two copies of it.

    The remedy is per operator *kind*, so repeating it once per node would make a wide plan's
    warning unreadable without saying anything the first copy did not.
    """
    both = _stream().distinct().union(_stream().distinct())
    (message,) = _warnings_for(both)
    assert message.count("distinct() keeps one entry per distinct row") == 1


def test_two_different_leaking_kinds_are_both_named():
    """Deduplicating by kind must not collapse *different* kinds into one."""
    plan = _stream().distinct().union(_stream().sort("k"))
    (message,) = _warnings_for(plan)
    assert "distinct() keeps one entry" in message
    assert "a full sort buffers the whole stream" in message


# --- the shapes that are bounded, and must stay silent -------------------------------


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("stateless", lambda d: d.filter(col("v") > 0).select("k")),
        (
            "watermarked_window",
            lambda d: (
                d.with_watermark("ts", "5m")
                .group_by(w=bt.window(col("ts"), "1h"))
                .agg(n=bt.count())
            ),
        ),
        (
            "watermark_dedup",
            lambda d: d.drop_duplicates_within_watermark(["k"], event_time="ts", lateness="5m"),
        ),
        ("topn", lambda d: d.sort("k").limit(5)),
        ("distinct_limit", lambda d: d.distinct().limit(5)),
        ("limit", lambda d: d.limit(5)),
    ],
)
def test_a_bounded_shape_is_silent(label, build):
    """A warning that fires on `head(10)` is one nobody reads on the query that does leak.

    `topn` and `distinct_limit` are the two that matter: their *node* retains unbounded
    state and their *plan* does not, because the limit above caps it — which is exactly
    what the engine's fusion rules encode by giving `Sort` and `Distinct` a `limit` field.
    """
    assert _warnings_for(build(_stream())) == [], label


def test_a_bounded_source_is_silent_however_stateful_the_plan():
    """A bounded input ends, and end-of-input releases every operator's state.

    The same node that leaks over a topic is an ordinary breaker over a file, which is why
    this cannot be a plan-build-time check and why no bounded test can see the problem.
    """
    assert _warnings_for(_bounded().group_by("k").agg(n=bt.count())) == []
    assert _warnings_for(_bounded().distinct()) == []
    assert _warnings_for(_bounded().sort("k")) == []


def test_a_filter_between_the_limit_and_the_breaker_does_not_cap_it():
    """Only the *direct* input of a limit is discounted, and deliberately so.

    A `Filter` under a `Limit` does not bound what the operator below the filter retains:
    the dedup still sees every row and still holds every distinct one, however few survive
    the filter. Widening the discount to "anywhere below a limit" would silence a real leak.
    """
    assert _warnings_for(_stream().distinct().filter(col("k") > 0).limit(5)) != []


# --- explain() answers the streaming questions too -----------------------------------


def test_explain_reports_what_a_streaming_plan_will_do():
    """Every number in an `explain()` of a streaming plan is a placeholder.

    The row estimate is the `unknown_rows` sentinel with `DEFAULT` provenance, which a
    stream shares with any bounded source whose size merely could not be measured. So the
    rendering said ``est≈1,000,000,000,000 (default)`` and nothing about the two things that
    decide whether the query works. Both were already computed and neither reached the
    reader.
    """
    out = _stream().group_by("k").agg(n=bt.count()).explain()
    assert "unbounded source" in out
    assert "emits incrementally" in out
    assert "retains state nothing releases: aggregate" in out


def test_explain_names_the_blocking_operator_for_a_plan_that_cannot_stream():
    out = _stream().sort("k").explain()
    assert "cannot emit until the input ends: sort" in out
    assert "will not stream" in out


def test_explain_of_a_bounded_plan_is_unchanged():
    """The questions do not arise for a bounded input, so nothing may be added to its plan."""
    out = _bounded().group_by("k").agg(n=bt.count()).explain()
    assert "streaming" not in out
    assert "unbounded" not in out


def test_explain_reports_a_streaming_udf_pipeline_too():
    """A `map_batches` pipeline over a topic is the shape most likely to be a stream, and it
    takes the branch that cannot lower to IR — so it is the one that needs this most."""
    out = _stream().map_batches(lambda b: b).explain()
    assert "unbounded source" in out
    assert "emits incrementally" in out


# --- the launcher actually asks ------------------------------------------------------


def test_starting_a_streaming_query_emits_the_warning():
    """The analysis had no caller for its whole life; this is the test that it has one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        query = (
            _stream()
            .group_by("k")
            .agg(n=bt.count())
            .write.memory(
                "state_warning_demo", trigger=bt.Trigger.available_now(), output_mode="complete"
            )
        )
        query.await_termination()
    leaks = [w for w in caught if "retains state that nothing releases" in str(w.message)]
    assert leaks, [str(w.message) for w in caught]
