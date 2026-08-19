"""Public surface with no test: selector introspection, progress records, errors, config.

Seven exported names had no test of any kind. They are small, and each one is small in a
way that hides a real failure:

* ``Selector.matched_columns`` / ``output_name`` are the introspection a user calls to see
  what a selector will do *before* running a query. If they disagreed with what the
  projection actually does, the preview would be a lie.
* ``SourceProgress`` / ``SinkProgress`` are the per-micro-batch records a streaming
  listener reads. They exist for Spark parity, so their field names are the contract.
* ``StreamingQueryListener.onQueryStarted`` / ``onQueryTerminated`` are the camelCase
  aliases that same parity requires, and an alias that does not forward is invisible.
* ``CompileError``, ``OptimizationError`` and ``TransportError`` are public exception
  types; what matters about them is where they sit in the hierarchy, because that is what
  decides whether a user's ``except`` clause catches them.
* ``QuarantineConfig`` is a tunable dataclass whose defaults are the shipped policy.
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

COLUMNS = {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "s": ["x", "y", "z"], "flag": [True, False, True]}


@pytest.fixture
def ds():
    return bt.from_pydict(COLUMNS)


@pytest.fixture
def schema(ds):
    return SchemaRef.from_arrow(ds.schema)


def test_matched_columns_previews_exactly_what_the_projection_selects(ds, schema):
    """The preview and the query must agree, or the preview is worse than nothing."""
    names = list(COLUMNS)
    for selector, expected in [
        (bt.numeric(), ["a", "b"]),
        (bt.integer(), ["a"]),
        (bt.floating(), ["b"]),
        (bt.string(), ["s"]),
        (bt.boolean(), ["flag"]),
        (bt.matches("^[ab]$"), ["a", "b"]),
        (bt.exclude("s", "flag"), ["a", "b"]),
        (bt.all(), names),
    ]:
        previewed = selector.matched_columns(names, schema)
        assert previewed == expected, f"{selector!r} previewed {previewed}"
        selected = list(ds.select(selector).to_pydict())
        assert selected == expected, f"{selector!r} selected {selected}, previewed {previewed}"


def test_matched_columns_keeps_the_input_column_order(schema):
    """Documented as "in the input's column order", which set algebra could easily lose."""
    names = list(COLUMNS)
    assert (bt.string() | bt.integer()).matched_columns(names, schema) == ["a", "s"]
    assert (bt.integer() | bt.string()).matched_columns(names, schema) == ["a", "s"], (
        "the order of the union's operands must not change the output order"
    )


def test_the_selector_algebra_composes_the_way_set_operations_do(schema):
    """Union, intersection, difference and complement, each against its set answer."""
    names = list(COLUMNS)
    numeric = set(bt.numeric().matched_columns(names, schema))
    integer = set(bt.integer().matched_columns(names, schema))
    assert set((bt.numeric() | bt.string()).matched_columns(names, schema)) == numeric | {"s"}
    assert set((bt.numeric() & bt.integer()).matched_columns(names, schema)) == numeric & integer
    assert set((bt.numeric() - bt.integer()).matched_columns(names, schema)) == numeric - integer
    assert set((~bt.numeric()).matched_columns(names, schema)) == set(names) - numeric


def test_output_name_reports_the_rename_a_selector_applies(ds, schema):
    """Without a rename the name passes through; with one it is what the query produces."""
    plain = bt.numeric()
    assert plain.output_name("a") == "a"

    renamed = bt.numeric().name.suffix("_n")
    assert renamed.output_name("a") == "a_n"
    produced = list(ds.with_columns(renamed).to_pydict())
    for column in bt.numeric().matched_columns(list(COLUMNS), schema):
        assert renamed.output_name(column) in produced, (
            f"output_name promised {renamed.output_name(column)}, which the query did not make"
        )


def test_a_selector_matching_nothing_previews_an_empty_list(schema):
    """Empty, not an error and not every column -- the two ways this usually goes wrong."""
    assert bt.matches("^nothing_here$").matched_columns(list(COLUMNS), schema) == []
    assert (bt.numeric() & bt.string()).matched_columns(list(COLUMNS), schema) == []


def test_source_and_sink_progress_carry_the_spark_field_names():
    """These records exist for Spark parity, so the field names are the contract."""
    source = bt.SourceProgress(
        description="rate", num_input_rows=17, start_offset={"o": 0}, end_offset={"o": 17}
    )
    assert source.description == "rate"
    assert source.num_input_rows == 17
    assert source.start_offset == {"o": 0}
    assert source.end_offset == {"o": 17}

    sink = bt.SinkProgress(description="memory", num_output_rows=17, token="batch-3")
    assert sink.description == "memory"
    assert sink.num_output_rows == 17
    assert sink.token == "batch-3"


def test_progress_records_default_to_an_empty_batch():
    """A source that produced nothing must report zero, not null -- listeners sum these."""
    source = bt.SourceProgress(description="rate")
    sink = bt.SinkProgress(description="memory")
    assert source.num_input_rows == 0
    assert source.start_offset is None and source.end_offset is None
    assert sink.num_output_rows == 0
    assert sink.token is None
    assert dataclasses.is_dataclass(source) and dataclasses.is_dataclass(sink)


def test_either_spelling_of_a_listener_callback_is_dispatched():
    """A listener ported from PySpark overrides ``onQueryStarted`` and must still be called.

    The two spellings are not aliases that forward into each other -- both are no-op hooks
    on the base class and the dispatcher calls both, so overriding either one works. That
    is a deliberate design and it is invisible from the class alone: the property only
    holds because of what ``_fire`` does, so this test goes through the dispatcher.
    """
    from batcher.plan.streaming.listener import notify_query_started, notify_query_terminated

    seen: list[str] = []

    class SparkStyle(bt.StreamingQueryListener):
        def onQueryStarted(self, event):
            seen.append("spark-start")

        def onQueryTerminated(self, event):
            seen.append("spark-stop")

    class PythonStyle(bt.StreamingQueryListener):
        def on_query_started(self, event):
            seen.append("python-start")

        def on_query_terminated(self, event):
            seen.append("python-stop")

    ported, native = SparkStyle(), PythonStyle()
    bt.add_streaming_listener(ported)
    bt.add_streaming_listener(native)
    try:
        notify_query_started("q", 0.0)
        notify_query_terminated("q", None)
    finally:
        bt.remove_streaming_listener(ported)
        bt.remove_streaming_listener(native)

    assert "spark-start" in seen, "the PySpark spelling was never called"
    assert "python-start" in seen, "the Python spelling was never called"
    assert "spark-stop" in seen and "python-stop" in seen


def test_a_listener_that_raises_does_not_break_the_query():
    """Documented: callbacks run on the query loop, so one bad listener must not stop it."""
    from batcher.plan.streaming.listener import notify_query_started

    reached: list[str] = []

    class Broken(bt.StreamingQueryListener):
        def on_query_started(self, event):
            raise RuntimeError("listener blew up")

    class Fine(bt.StreamingQueryListener):
        def on_query_started(self, event):
            reached.append("ok")

    broken, fine = Broken(), Fine()
    bt.add_streaming_listener(broken)
    bt.add_streaming_listener(fine)
    try:
        notify_query_started("q", 0.0)
    finally:
        bt.remove_streaming_listener(broken)
        bt.remove_streaming_listener(fine)
    assert reached == ["ok"], "a raising listener stopped the ones after it"


def test_the_default_listener_callbacks_do_nothing_rather_than_raise():
    """Documented: a listener overriding one callback must not be broken by the others."""
    listener = bt.StreamingQueryListener()
    assert listener.onQueryStarted(object()) is None
    assert listener.onQueryTerminated(object()) is None
    assert listener.on_query_progress(object()) is None


def test_a_listener_registers_and_deregisters():
    """``add``/``remove`` round trip, so a test that adds one does not leak it."""
    listener = bt.StreamingQueryListener()
    bt.add_streaming_listener(listener)
    try:
        assert bt.remove_streaming_listener(listener) is True
    finally:
        bt.remove_streaming_listener(listener)
    assert bt.remove_streaming_listener(listener) is False, "removing twice reports false"


#: Each public error and the base a user's ``except`` clause would name.
ERROR_HIERARCHY = [
    (bt.CompileError, bt.ExecutionError),
    (bt.OptimizationError, bt.BatcherError),
    (bt.TransportError, bt.BatcherError),
]


@pytest.mark.parametrize(("error", "base"), ERROR_HIERARCHY)
def test_public_errors_sit_where_an_except_clause_expects_them(error, base):
    """The hierarchy is the contract: it decides what a user's handler catches."""
    assert issubclass(error, base), f"{error.__name__} is not a {base.__name__}"
    assert issubclass(error, bt.BatcherError), "every engine error is a BatcherError"
    assert issubclass(error, Exception)
    with pytest.raises(base):
        raise error("boom")
    with pytest.raises(bt.BatcherError):
        raise error("boom")


def test_compile_error_is_an_execution_error_because_the_interpreter_remains():
    """Its docstring says the JIT failing leaves a fallback, and the base encodes that.

    A ``CompileError`` reaching a user is an execution-time event, not a planning one, so a
    handler that catches ``OptimizationError`` must *not* catch it -- these two are the
    pair most likely to be conflated.
    """
    assert issubclass(bt.CompileError, bt.ExecutionError)
    assert not issubclass(bt.CompileError, bt.OptimizationError)
    assert not issubclass(bt.OptimizationError, bt.ExecutionError)


def test_quarantine_config_defaults_are_the_shipped_policy():
    """The defaults are what runs when nobody tunes anything, so they are pinned."""
    from batcher.config import QuarantineConfig

    config = QuarantineConfig()
    assert config.enabled is True
    assert config.failure_threshold == 3.0
    assert config.half_life_s == 300.0
    assert config.cooldown_s == 60.0
    assert config.max_cooldown_s == 900.0
    assert config.max_blocked_fraction == 0.34


def test_quarantine_config_cooldowns_and_fractions_are_coherent():
    """The invariants the fields have to satisfy for the policy to make sense."""
    from batcher.config import QuarantineConfig

    config = QuarantineConfig()
    assert config.cooldown_s <= config.max_cooldown_s, "the cap must not be below the start"
    assert 0.0 < config.max_blocked_fraction < 1.0, (
        "blocking every worker is never a recovery strategy"
    )
    assert config.half_life_s > 0.0 and config.failure_threshold > 0.0

    tuned = QuarantineConfig(enabled=False, cooldown_s=5.0, max_cooldown_s=10.0)
    assert tuned.enabled is False
    assert tuned.cooldown_s == 5.0
    assert tuned.failure_threshold == 3.0, "an untouched field keeps its default"


def test_quarantine_config_is_reachable_through_the_public_config_surface():
    """It is exported from ``batcher.config``, which is where a user would look for it."""
    import batcher.config as config

    assert "QuarantineConfig" in config.__all__
    assert config.QuarantineConfig is not None
