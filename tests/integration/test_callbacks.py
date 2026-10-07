"""Integration coverage for the Ray-Data-style callback transforms.

`map`/`flat_map`/`@udf` are black-box Python callbacks (no DuckDB oracle), routed
through the worker-side `map_batches` path. The per-row function runs in the worker
(data plane), never the driver.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import udf

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_pydict({"a": [1, 2, 3], "b": [10, 20, 30]})


def test_map_per_row_adds_column():
    out = (
        _ds().map(lambda r: {"a": r["a"], "b": r["b"], "c": r["a"] + r["b"]}).collect().to_pydict()
    )
    assert out == {"a": [1, 2, 3], "b": [10, 20, 30], "c": [11, 22, 33]}


def test_flat_map_one_to_many():
    out = (
        _ds()
        .flat_map(lambda r: [{"a": r["a"], "k": i} for i in range(r["a"])])
        .collect()
        .to_pydict()
    )
    assert out == {"a": [1, 2, 2, 3, 3, 3], "k": [0, 0, 1, 0, 1, 2]}


def test_flat_map_can_drop_rows():
    # Returning [] for a row drops it (a filtering flat_map).
    out = _ds().flat_map(lambda r: [r] if r["a"] % 2 else []).collect().to_pydict()
    assert out == {"a": [1, 3], "b": [10, 30]}


def test_udf_per_row_and_batch():
    @udf(per_row=True)
    def doubled(r):
        return {"a": r["a"], "d": r["a"] * 2}

    assert doubled(_ds()).collect().to_pydict() == {"a": [1, 2, 3], "d": [2, 4, 6]}

    @udf()
    def scale(batch):
        import pyarrow as pa

        return batch.append_column("x", pa.array([v.as_py() * 100 for v in batch.column("a")]))

    assert scale(_ds()).collect().to_pydict()["x"] == [100, 200, 300]


def test_udf_forwards_resilience_config():
    """`@udf(max_retries=...)` forwards the resilience knobs to `map_batches`, so a decorated
    transform retries a transient failure like any other."""
    state = {"n": 0}

    @udf(max_retries=3, retry_backoff=0.0)
    def flaky(batch):
        state["n"] += 1
        if state["n"] < 2:
            raise ConnectionError("transient")
        return batch

    assert flaky(_ds()).collect().to_pydict()["a"] == [1, 2, 3]
    assert state["n"] == 2


def test_udf_composes_with_async():
    """`@udf` over an async fn routes through the concurrent event-loop path."""
    import asyncio

    @udf(max_concurrency=4)
    async def enrich(batch):
        await asyncio.sleep(0.001)
        return batch.append_column(
            "c", __import__("pyarrow").array([v.as_py() + 1 for v in batch.column("a")])
        )

    assert enrich(_ds()).collect().to_pydict()["c"] == [2, 3, 4]


# --- the shared failure policy on the row verbs (AP-361) --------------------------------

_FAILURE_DEFAULTS = {
    "max_errored_rows": 0,
    "error_column": None,
    "timeout": 0.0,
    "max_retries": 0,
    "retry_backoff": 0.5,
    "retry_on": None,
}


@pytest.mark.parametrize("verb", ["map_batches", "map", "flat_map", "filter"])
def test_every_callback_verb_takes_the_same_failure_policy(verb: str) -> None:
    import inspect

    params = inspect.signature(getattr(bt.Dataset, verb)).parameters
    assert {name: params[name].default for name in _FAILURE_DEFAULTS} == _FAILURE_DEFAULTS


def _flaky_rows(fail_times: int):
    """A row fn that raises `ConnectionError` on its first `fail_times` calls."""
    state = {"calls": 0}

    def fn(row):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise ConnectionError("transient")
        return row

    return fn, state


@pytest.mark.parametrize("verb", ["map", "flat_map"])
def test_a_row_verb_retries_a_transient_failure(verb: str) -> None:
    fn, _ = _flaky_rows(1)
    wrapped = fn if verb == "map" else (lambda row: [fn(row)])
    out = getattr(_ds(), verb)(wrapped, max_retries=2, retry_backoff=0.0, num_workers=1)
    assert out.collect().to_pydict() == {"a": [1, 2, 3], "b": [10, 20, 30]}


def test_a_row_verb_without_retries_still_fails_fast() -> None:
    fn, _ = _flaky_rows(1)
    with pytest.raises(ConnectionError):
        _ds().map(fn, num_workers=1).collect()


def test_retry_on_restricts_what_a_row_verb_retries() -> None:
    fn, state = _flaky_rows(1)
    with pytest.raises(ConnectionError):
        _ds().map(fn, max_retries=3, retry_backoff=0.0, retry_on=KeyError).collect()
    assert state["calls"] == 1


def test_a_callable_filter_retries_a_transient_failure() -> None:
    state = {"calls": 0}

    def keep(batch):
        state["calls"] += 1
        if state["calls"] == 1:
            raise ConnectionError("transient")
        import pyarrow.compute as pc

        return pc.greater(batch.column("a"), 1)

    out = _ds().filter(keep, max_retries=1, retry_backoff=0.0, num_workers=1)
    assert out.collect().to_pydict() == {"a": [2, 3], "b": [20, 30]}


def test_a_row_verb_times_out_a_hung_call() -> None:
    import time

    def slow(row):
        time.sleep(2.0)
        return row

    with pytest.raises(TimeoutError):
        _ds().map(slow, timeout=0.05, num_workers=1).collect()


def test_per_row_udf_forwards_resilience_config() -> None:
    fn, state = _flaky_rows(1)
    decorated = udf(per_row=True, max_retries=2, retry_backoff=0.0)(fn)
    assert decorated(_ds()).collect().to_pydict()["a"] == [1, 2, 3]
    assert state["calls"] > 1


def test_retry_options_on_an_expression_filter_are_refused() -> None:
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="max_retries"):
        _ds().filter(bt.col("a") > 1, max_retries=2)


# --- keeping the rows max_errored_rows would drop (AP-370) ------------------------------


def _bad_on_multiples_of_two(row):
    if row["a"] % 2 == 0:
        raise ValueError(f"bad {row['a']}")
    return {"y": row["a"] * 10}


_Y = __import__("pyarrow").schema([("y", __import__("pyarrow").int64())])


@pytest.mark.parametrize("num_workers", [1, 4])
def test_error_column_keeps_a_failing_row_with_its_error(num_workers: int) -> None:
    ds = bt.from_pydict({"a": list(range(1, 9))}).map(
        _bad_on_multiples_of_two,
        output_columns=_Y,
        # The allowance is shared by every run of this `fn` in the process, and this test
        # runs the stage three times, so it is sized for all of them.
        max_errored_rows=100,
        error_column="err",
        batch_size=3,
        num_workers=num_workers,
    )
    out = ds.collect()
    assert out.column("y").to_pylist() == [10, None, 30, None, 50, None, 70, None]
    errors = [None, "ValueError: bad 2", None, "ValueError: bad 4"]
    assert out.column("err").to_pylist() == [
        *errors,
        None,
        "ValueError: bad 6",
        None,
        "ValueError: bad 8",
    ]
    assert ds.schema == out.schema
    quarantined = ds.filter(bt.col("err").is_not_null()).count()
    assert quarantined == 4


def test_error_column_on_flat_map() -> None:
    ds = bt.from_pydict({"a": [1, 2, 3]}).flat_map(
        lambda r: [_bad_on_multiples_of_two(r)] * 2,
        output_columns=_Y,
        max_errored_rows=5,
        error_column="err",
    )
    assert ds.collect().to_pydict() == {
        "y": [10, 10, None, 30, 30],
        "err": [None, None, "ValueError: bad 2", None, None],
    }


def test_error_column_carries_preserved_columns_on_map_batches() -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    def fn(batch):
        if pc.any(pc.equal(batch.column("a"), 2)).as_py():
            raise ValueError("two")
        return batch.append_column("y", pc.multiply(batch.column("a"), 10))

    ds = bt.from_pydict({"a": [1, 2, 3]}).map_batches(
        fn,
        preserves_columns=["a"],
        output_columns=pa.schema([("a", pa.int64()), ("y", pa.int64())]),
        max_errored_rows=1,
        error_column="err",
    )
    assert ds.collect().to_pydict() == {
        "a": [1, 2, 3],
        "y": [10, None, 30],
        "err": [None, "ValueError: two", None],
    }


def test_error_column_on_a_callable_filter_keeps_the_input_row() -> None:
    def keep(batch):
        values = batch.column("a").to_pylist()
        return [10 // (v - 2) > 0 for v in values]

    out = _ds().filter(keep, max_errored_rows=1, error_column="err").collect().to_pydict()
    assert out == {
        "a": [2, 3],
        "b": [20, 30],
        "err": ["ZeroDivisionError: integer division or modulo by zero", None],
    }


def test_error_column_on_an_async_callback() -> None:
    import asyncio

    import pyarrow as pa

    async def fn(batch):
        await asyncio.sleep(0)
        if 2 in batch.column("a").to_pylist():
            raise ValueError("two")
        return {"y": batch.column("a")}

    ds = _ds().map_batches(
        fn, output_columns=pa.schema([("y", pa.int64())]), max_errored_rows=1, error_column="e"
    )
    assert ds.collect().to_pydict() == {"y": [1, None, 3], "e": [None, "ValueError: two", None]}


def test_kept_rows_still_spend_the_budget() -> None:
    def bad_on_evens(row):  # its own function, so no other test has drawn on its allowance
        return _bad_on_multiples_of_two(row)

    ds = bt.from_pydict({"a": list(range(1, 9))}).map(
        bad_on_evens, output_columns=_Y, max_errored_rows=2, error_column="err"
    )
    with pytest.raises(ValueError, match="bad"):
        ds.collect()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"error_column": "err"}, "max_errored_rows > 0"),
        ({"error_column": "err", "max_errored_rows": 1, "output_columns": ["y"]}, "Schema"),
        ({"error_column": "y", "max_errored_rows": 1, "output_columns": _Y}, "already"),
        ({"error_column": "", "max_errored_rows": 1, "output_columns": _Y}, "non-empty"),
    ],
)
def test_an_error_column_the_stage_cannot_honour_is_refused(kwargs, message) -> None:
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match=message):
        _ds().map(_bad_on_multiples_of_two, **kwargs)


# --- a registered SQL function named from an expression (AP-375) ------------------------


def test_call_function_on_a_registered_function_says_where_it_works() -> None:
    import pyarrow.compute as pc

    from batcher._internal.errors import PlanError

    s = bt.Session()
    s.register_function("dbl_cb", lambda a: pc.multiply(a, 2))
    with s.activate(), pytest.raises(PlanError) as info:
        bt.call_function("dbl_cb", "a")
    message = str(info.value)
    assert "callable only inside a SQL query" in message
    assert "is not registered" not in message  # it is registered; that claim was false
    s.register("t", _ds())
    assert s.sql("SELECT dbl_cb(a) AS d FROM t").to_pydict() == {"d": [2, 4, 6]}


def test_an_unknown_function_in_a_query_still_points_at_registration() -> None:
    from batcher._internal.errors import SQLUnsupportedError

    with pytest.raises(SQLUnsupportedError, match=r"bt\.register_function"):
        bt.sql("SELECT nope_cb(a) FROM t", t=_ds()).collect()
