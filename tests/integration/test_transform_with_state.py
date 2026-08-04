"""Arbitrary keyed state over a stream — the escape hatch the relational operators lack.

Sessionization with custom rules, a running fraud score, a per-device state machine: none
of those has an algebra, so none of them is expressible as an aggregate. Spark grew
`mapGroupsWithState` / `transformWithState` for exactly this, and Batcher had nothing.

The properties worth pinning are the ones that make it an *operator* rather than a
callback: the same rows whether it is collected or iterated, state that survives a
restart, state that a TTL bounds, and a distributed request that is refused rather than
quietly answered by one machine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError, ResourceError

_SCHEMA = pa.schema([("user", pa.string()), ("v", pa.int64())])


def _running_total(key, rows, state):
    total = (state or {"total": 0})["total"] + sum(rows.column("v").to_pylist())
    return {"user": [key[0]], "total": [total]}, {"total": total}


def _feed(*batches):
    def gen():
        for users, values in batches:
            yield pa.record_batch({"user": list(users), "v": list(values)}, schema=_SCHEMA)

    return gen


def _stream(*batches):
    return bt.from_batches(_feed(*batches), _SCHEMA, bounded=False)


def _with_state(dataset, **kwargs):
    options = {
        "group_by": "user",
        "output_columns": ["user", "total"],
        "state_ttl": "1 hour",
        **kwargs,
    }
    return dataset.transform_with_state(_running_total, **options)


def _pairs(table) -> list[tuple[str, int]]:
    data = table.to_pydict()
    return sorted(zip(data["user"], data["total"], strict=True))


@pytest.mark.integration
def test_a_bounded_input_folds_each_key_once():
    events = bt.from_pydict({"user": ["a", "b", "a"], "v": [1, 2, 3]})
    assert _pairs(_with_state(events).collect()) == [("a", 4), ("b", 2)]


@pytest.mark.integration
def test_a_stream_calls_the_function_once_per_key_per_micro_batch():
    """The state is what carries a key across batches, so the running total advances."""
    stream = _stream((["a", "b"], [1, 2]), (["a"], [3]), (["b", "a"], [4, 5]))
    rows: list[dict] = []
    for batch in _with_state(stream).iter_batches():
        rows.extend(batch.to_pylist())
    assert [(r["user"], r["total"]) for r in rows] == [
        ("a", 1),
        ("b", 2),
        ("a", 4),
        ("a", 9),
        ("b", 6),
    ]


@pytest.mark.integration
def test_a_key_removed_by_returning_no_state_starts_over():
    def reset_after_two(key, rows, state):
        seen = (state or {"n": 0})["n"] + rows.num_rows
        if seen >= 2:
            return {"user": [key[0]], "total": [seen]}, None
        return {"user": [key[0]], "total": [seen]}, {"n": seen}

    stream = _stream((["a"], [1]), (["a"], [1]), (["a"], [1]))
    rows: list[dict] = []
    for batch in stream.transform_with_state(
        reset_after_two, group_by="user", output_columns=["user", "total"]
    ).iter_batches():
        rows.extend(batch.to_pylist())
    assert [r["total"] for r in rows] == [1, 2, 1], "the key was not forgotten"


@pytest.mark.integration
def test_a_function_that_emits_nothing_emits_nothing():
    def silent(key, rows, state):
        return None, {"n": rows.num_rows}

    stream = _stream((["a"], [1]))
    assert (
        list(
            stream.transform_with_state(
                silent, group_by="user", output_columns=["user"]
            ).iter_batches()
        )
        == []
    )


@pytest.mark.integration
def test_it_writes_to_a_sink_with_the_rows_iter_batches_yields():
    """One fold, two terminals — not a second definition of the operator."""
    iterated: list[tuple[str, int]] = []
    for batch in _with_state(_stream((["a", "b"], [1, 2]), (["a"], [3]))).iter_batches():
        iterated.extend((r["user"], r["total"]) for r in batch.to_pylist())

    query = _with_state(_stream((["a", "b"], [1, 2]), (["a"], [3]))).write.memory(
        "tws_sink", trigger=bt.Trigger.available_now()
    )
    assert query.await_termination(timeout=60) is True
    assert query.exception() is None
    assert _pairs(bt.read_memory("tws_sink").collect()) == sorted(iterated)


@pytest.mark.integration
def test_the_progress_record_reports_the_retained_key_count():
    query = _with_state(_stream((["a", "b"], [1, 2]))).write.memory(
        "tws_metrics", trigger=bt.Trigger.available_now()
    )
    assert query.await_termination(timeout=60) is True
    operators = query.recent_progress[0].state_operators
    assert len(operators) == 1
    assert operators[0].operator_name == "transform_with_state"
    assert operators[0].num_rows_total == 2, "two keys were seen"


@pytest.mark.integration
def test_state_survives_a_restart_through_the_checkpoint(tmp_path):
    """A stateful operator that cannot resume is a demo, so this is the load-bearing one."""

    def counter(key, rows, state):
        n = (state or {"n": 0})["n"] + rows.num_rows
        return {"bucket": [key[0]], "n": [n]}, {"n": n}

    checkpoint = str(tmp_path / "ckpt")

    def run(name: str, num_rows: int):
        stateful = (
            bt.read.rate(5, num_rows=num_rows, pace=False)
            .with_columns(bucket=bt.col("value") % 2)
            .transform_with_state(
                counter, group_by="bucket", output_columns=["bucket", "n"], state_ttl="1 hour"
            )
        )
        query = stateful.write.memory(
            name, trigger=bt.Trigger.available_now(), checkpoint=checkpoint, query_name="tws-ck"
        )
        assert query.await_termination(timeout=60) is True
        assert query.exception() is None
        return bt.read_memory(name).to_pydict()

    first = run("tws_ck1", 10)
    assert max(first["n"]) == 5

    second = run("tws_ck2", 20)
    assert min(second["n"]) > 5, "the restart began from an empty state"


@pytest.mark.integration
def test_a_ttl_forgets_a_key_that_went_quiet():
    """Without it the operator retains a row per key seen since the query started."""
    from batcher.core.streaming import KeyedStateFold

    fold = KeyedStateFold(
        _with_state(bt.from_pydict({"user": ["a"], "v": [1]}), state_ttl="1 hour")._plan
    )
    fold.push(pa.record_batch({"user": ["a", "b"], "v": [1, 2]}, schema=_SCHEMA))
    assert fold.metrics().num_rows_total == 2

    fold._ttl = 1  # one microsecond: everything already seen is now stale
    fold.push(pa.record_batch({"user": ["c"], "v": [3]}, schema=_SCHEMA))
    assert fold.metrics().num_rows_total == 0
    assert fold.metrics().num_rows_removed >= 2


@pytest.mark.integration
def test_state_that_cannot_be_checkpointed_is_refused_by_key():
    """Failing here rather than at the next snapshot is the difference between a message
    naming the key and an ArrowInvalid from inside the checkpoint writer an hour later."""

    def bad(key, rows, state):
        return None, {"payload": {"nested": 1}}

    stream = _stream((["a"], [1]))
    with pytest.raises(PlanError, match="must be scalars"):
        list(
            stream.transform_with_state(
                bad, group_by="user", output_columns=["user"]
            ).iter_batches()
        )


@pytest.mark.integration
def test_a_non_mapping_state_is_refused_too():
    def bad(key, rows, state):
        return None, [1, 2, 3]

    stream = _stream((["a"], [1]))
    with pytest.raises(PlanError, match="flat mapping of scalars"):
        list(
            stream.transform_with_state(
                bad, group_by="user", output_columns=["user"]
            ).iter_batches()
        )


@pytest.mark.integration
def test_unbounded_state_fails_loudly_rather_than_growing():
    from batcher.config import Config, MemoryConfig, config_context

    tiny = Config().replace(memory=MemoryConfig(streaming_state_max_bytes=1))
    with config_context(tiny), pytest.raises(ResourceError, match="transform_with_state retained"):
        list(_with_state(_stream((["a", "b"], [1, 2])), state_ttl=None).iter_batches())


@pytest.mark.integration
def test_a_distributed_request_is_refused_rather_than_run_on_one_machine():
    events = bt.from_pydict({"user": ["a"], "v": [1]})
    with pytest.raises(PlanError, match="no distributed implementation"):
        _with_state(events).collect(distributed=True)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_by": []}, "at least one column"),
        ({"group_by": "nope"}, "unknown group_by"),
        ({"output_columns": []}, "output_columns must name"),
    ],
)
def test_an_impossible_declaration_is_refused_at_build_time(kwargs, message):
    events = bt.from_pydict({"user": ["a"], "v": [1]})
    with pytest.raises(PlanError, match=message):
        _with_state(events, **kwargs)


@pytest.mark.integration
def test_an_aggregating_output_mode_is_refused():
    with pytest.raises(PlanError, match="needs an aggregation"):
        _with_state(_stream((["a"], [1]))).write.memory(
            "tws_mode", trigger=bt.Trigger.available_now(), output_mode="complete"
        )


@pytest.mark.integration
def test_a_composite_key_is_passed_through_in_declaration_order():
    seen: list[tuple] = []

    def record(key, rows, state):
        seen.append(key)
        return None, {"n": 1}

    events = bt.from_pydict({"a": ["x", "x"], "b": [1, 2], "v": [1, 1]})
    events.transform_with_state(record, group_by=["a", "b"], output_columns=["a"]).collect()
    assert sorted(seen) == [("x", 1), ("x", 2)]


@pytest.mark.integration
def test_null_keys_group_together_rather_than_one_group_per_row():
    """`NULL = NULL` is null in SQL, but *grouping* treats two nulls as one group — and the
    boundary test has to say so explicitly or every null-keyed row becomes its own."""
    seen: list[tuple] = []

    def record(key, rows, state):
        seen.append((key, rows.num_rows))
        return None, {"n": 1}

    events = bt.from_pydict({"user": [None, None, "a"], "v": [1, 2, 3]})
    events.transform_with_state(record, group_by="user", output_columns=["user"]).collect()
    assert dict(seen) == {(None,): 2, ("a",): 1}, "the two null-keyed rows split into groups"
