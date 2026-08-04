"""The Spark-parity surface: one starting-position vocabulary, the dev sources, the
benchmark sink, and the identifiers a supervisor keys on.

Each of these is small on its own. Together they are the difference between a PySpark job
that ports by changing `spark.readStream` to `bt.read` and one that needs its options
looked up connector by connector — five brokers had five spellings of "start at the end",
`rate` could not produce a reproducible batch size, there was no sink to benchmark
against, and a restart was indistinguishable from a long-running query.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.broker.schema import normalize_starting_position
from batcher.io.formats.streaming.dev import RateMicroBatchSource, SocketSource
from batcher.io.formats.streaming.eventhubs import EventHubsSource
from batcher.io.formats.streaming.kinesis import KinesisSource
from batcher.io.formats.streaming.pulsar import PulsarSource
from batcher.io.formats.streaming.sinks import STREAM_SINKS, NoopStreamSink
from batcher.plan.streaming import StateOperatorProgress, StreamingQueryProgress

# --- one starting-position vocabulary -------------------------------------------------

_ALIASES = {"earliest": "TRIM_HORIZON", "latest": "LATEST"}


@pytest.mark.parametrize(("asked", "native"), [("earliest", "TRIM_HORIZON"), ("latest", "LATEST")])
def test_a_shared_name_maps_to_the_brokers_own_spelling(asked, native):
    assert normalize_starting_position(asked, aliases=_ALIASES) == native


def test_a_brokers_native_spelling_still_works():
    """Refusing it would break every reader that already passes one."""
    assert normalize_starting_position("LATEST", aliases=_ALIASES) == "LATEST"


def test_an_unknown_position_is_refused_with_both_vocabularies_listed():
    with pytest.raises(PlanError, match="unknown starting_position"):
        normalize_starting_position("erliest", aliases=_ALIASES)


def test_a_non_string_position_is_refused_by_type():
    with pytest.raises(PlanError, match="must be a string"):
        normalize_starting_position(3, aliases=_ALIASES)


def test_kinesis_takes_the_shared_name_and_its_own():
    assert KinesisSource("s", starting_position="latest")._options["iterator_type"] == "LATEST"
    assert KinesisSource("s", iterator_type="LATEST")._options["iterator_type"] == "LATEST"


def test_pulsar_takes_the_shared_name():
    assert PulsarSource("t", starting_position="latest")._starting_position == "Latest"
    assert PulsarSource("t")._starting_position == "Earliest"


def test_event_hubs_maps_onto_its_offset_sentinels():
    assert (
        EventHubsSource("h", starting_position="latest")._options["starting_position"] == "@latest"
    )
    assert EventHubsSource("h")._options["starting_position"] == "-1"


def test_event_hubs_passes_an_explicit_offset_straight_through():
    """Resuming from a recorded position is the other reason to set this at all."""
    assert EventHubsSource("h", starting_position="12345")._options["starting_position"] == "12345"


# --- rate_micro_batch -----------------------------------------------------------------


def test_rate_micro_batch_produces_a_fixed_number_of_rows_per_batch():
    """`rate` promises rows per *second*, so how many land in a batch depends on how long
    the last one took — which makes it useless as a benchmark input."""
    source = RateMicroBatchSource(3, num_rows=9)
    assert [b.num_rows for b in source.iter_batches()] == [3, 3, 3]


def test_its_event_time_advances_once_per_batch_not_once_per_row():
    """So a fixed number of batches always closes the same windows."""
    source = RateMicroBatchSource(2, num_rows=6, advance_ms_per_batch=1000)
    stamps = [b.column("timestamp").to_pylist() for b in source.iter_batches()]
    assert all(len(set(batch)) == 1 for batch in stamps), "rows within a batch differ in time"
    assert len({batch[0] for batch in stamps}) == 3, "batches did not advance event time"


def test_the_reader_exposes_it_under_the_industry_name():
    assert [b.num_rows for b in bt.read.rate_micro_batch(4, num_rows=8).iter_batches()] == [4, 4]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"rows_per_batch": 0}, "rows_per_batch"), ({"advance_ms_per_batch": -1}, "advance_ms")],
)
def test_an_impossible_rate_micro_batch_is_refused(kwargs, message):
    with pytest.raises(PlanError, match=message):
        RateMicroBatchSource(**{"rows_per_batch": 1, **kwargs})


def test_an_unbounded_rate_micro_batch_refuses_to_materialize():
    with pytest.raises(PlanError, match="unbounded"):
        RateMicroBatchSource(2).read()


# --- socket includeTimestamp ----------------------------------------------------------


def test_the_socket_source_can_produce_sparks_one_column_shape():
    assert SocketSource().schema().names == ["value", "timestamp"]
    assert SocketSource(include_timestamp=False).schema().names == ["value"]


# --- the noop benchmark sink ----------------------------------------------------------


def test_the_noop_sink_is_registered_under_its_industry_name():
    assert STREAM_SINKS.get("noop") is NoopStreamSink


def test_it_discards_the_rows_but_not_the_count():
    """A benchmark sink that swallowed the count would make "this is fast" and "this
    produced nothing" look identical."""
    sink = NoopStreamSink()
    sink.open()
    assert sink.write_batch(0, pa.table({"a": [1, 2, 3]})) == "noop:0:3"
    sink.write_batch(1, pa.table({"a": [4]}))
    assert (sink.rows_written, sink.batches_written) == (4, 2)


def test_reopening_it_resets_the_counters():
    sink = NoopStreamSink()
    sink.open()
    sink.write_batch(0, pa.table({"a": [1]}))
    sink.open()
    assert sink.rows_written == 0


@pytest.mark.integration
def test_the_writer_runs_a_whole_pipeline_into_it():
    query = bt.read.rate_micro_batch(100, num_rows=500).write.noop(
        trigger=bt.Trigger.available_now()
    )
    assert query.await_termination(timeout=60) is True
    assert sum(p.num_input_rows for p in query.recent_progress) == 500


# --- query identifiers and supervisor semantics ---------------------------------------


@pytest.mark.integration
def test_id_is_stable_across_runs_and_run_id_is_not():
    """A metrics system keyed only on `id` cannot tell a query that has been up for a week
    from one that has crash-looped every ten minutes."""
    schema = pa.schema([("v", pa.int64())])

    def feed():
        yield pa.record_batch({"v": [1]}, schema=schema)

    def run(name: str):
        query = bt.from_batches(feed, schema, bounded=False).write.memory(
            name, trigger=bt.Trigger.available_now(), query_name="supervised"
        )
        query.await_termination(timeout=30)
        return query

    first, second = run("rid1"), run("rid2")
    assert first.id == second.id == "supervised"
    assert first.run_id != second.run_id
    assert first.runId == first.run_id


@pytest.mark.integration
def test_reset_terminated_clears_a_reported_termination():
    """Without it a supervisor that restarts a failed query and loops back into
    `await_any_termination` spins at full speed on a termination it already handled."""
    bt.reset_terminated()
    assert bt.await_any_termination(timeout=0.0) is True


# --- progress serialization -----------------------------------------------------------


def test_progress_serializes_in_sparks_shape():
    progress = StreamingQueryProgress(
        3,
        10,
        8,
        5.0,
        0.0,
        name="q",
        state_operators=(StateOperatorProgress("windowed_aggregate", num_late_inputs_dropped=2),),
        duration_breakdown_ms=(("addBatch", 4.0),),
    )
    data = json.loads(progress.json())
    assert data["batchId"] == 3
    assert data["numLateRows"] == 2
    assert data["durationMs"]["addBatch"] == 4.0
    assert data["durationMs"]["triggerExecution"] == 5.0
    assert data["stateOperators"][0]["numRowsDroppedByWatermark"] == 2


def test_the_state_operator_carries_sparks_name_for_the_late_counter():
    operator = StateOperatorProgress("dedup", num_late_inputs_dropped=7)
    assert operator.num_rows_dropped_by_watermark == 7


def test_processed_rows_per_second_is_sparks_name_for_the_output_rate():
    progress = StreamingQueryProgress(0, 500, 50, 250.0, 0.0)
    assert progress.processed_rows_per_second == progress.output_rows_per_second == 200.0


@pytest.mark.integration
def test_a_real_query_reports_where_its_time_went():
    """A total alone cannot distinguish a slow query from a slow *checkpoint*, and the two
    have opposite remedies."""
    query = bt.read.rate_micro_batch(50, num_rows=100).write.noop(
        trigger=bt.Trigger.available_now()
    )
    assert query.await_termination(timeout=60) is True
    breakdown = query.recent_progress[-1].duration_ms_map
    assert set(breakdown) == {"latestOffset", "addBatch", "walCommit"}
    assert all(v >= 0 for v in breakdown.values())


# --- the plan a running query is running -----------------------------------------------


@pytest.mark.integration
def test_a_query_can_explain_the_plan_it_is_running():
    """A stream is the case where you most want the plan and least want to re-derive it:
    the Dataset that started it is often long out of scope, and `explain(analyze=True)`
    is not available because measuring would mean consuming the source a second time."""
    query = bt.read.rate_micro_batch(10, num_rows=20).write.noop(trigger=bt.Trigger.available_now())
    plan = query.explain()
    assert "scan" in plan
    assert query.await_termination(timeout=60) is True


@pytest.mark.integration
def test_explain_also_renders_json():
    query = bt.read.rate_micro_batch(10, num_rows=20).write.noop(trigger=bt.Trigger.available_now())
    assert json.loads(query.explain(format="json"))
    assert query.await_termination(timeout=60) is True


@pytest.mark.integration
def test_process_all_available_has_the_snake_case_spelling_too():
    """Every other Spark name on this handle aliases a snake_case primary. This one was
    only reachable in camelCase, so a pipeline written in Batcher's own idiom had to
    switch spelling for one call."""
    query = bt.read.rate_micro_batch(5, num_rows=10).write.noop(trigger=bt.Trigger.available_now())
    assert query.process_all_available() is True
    assert query.processAllAvailable() is True
