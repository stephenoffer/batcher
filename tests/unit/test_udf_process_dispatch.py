"""What a `map_batches` UDF may *be* if it is to run across processes, and what it costs.

Three separate claims are pinned here, all about the boundary between the driver and a UDF
worker process:

* a lambda or a closure — the most common spelling of a UDF, and the one plain `pickle`
  refuses — can cross, so a GIL-bound body is not confined to one core by its spelling;
* the `max_errored_rows` allowance means the same thing on the process path as on the thread
  path, so a corrupt row's fate does not depend on a scheduling decision the user never made;
* two locally-defined UDFs are two UDFs, and do not share one cached measurement or one
  error allowance because they happen to share a qualname.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.udf import processes, strategy
from batcher.plan.logical import MapBatches, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("x", pa.int64())])

_needs_cloudpickle = pytest.mark.skipif(
    processes._cloudpickle() is None, reason="cloudpickle is not installed"
)


def _op(fn, **kwargs) -> MapBatches:
    return MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=fn, **kwargs)


def _double(batch: pa.RecordBatch) -> pa.RecordBatch:
    import pyarrow.compute as pc

    return pa.record_batch({"x": pc.multiply(batch.column("x"), 2)})


# --- what can cross ---------------------------------------------------------------------


def test_a_module_level_function_crosses_by_reference():
    """The cheap case stays cheap: `pickle` names it, nothing is sent by value."""
    wire = processes.dispatchable(_double)
    assert wire is _double


@_needs_cloudpickle
def test_a_lambda_can_cross():
    wire = processes.dispatchable(lambda batch: batch)
    assert isinstance(wire, processes._ByValueFn)


@_needs_cloudpickle
def test_a_closure_can_cross_and_still_computes_the_same_thing():
    factor = 3

    def scale(batch: pa.RecordBatch) -> pa.RecordBatch:
        import pyarrow.compute as pc

        return pa.record_batch({"x": pc.multiply(batch.column("x"), factor)})

    wire = processes.dispatchable(scale)
    batch = pa.record_batch({"x": pa.array([1, 2, 3])})
    # Locally it is the original callable; over the wire it round-trips to an equal one.
    assert wire(batch).column("x").to_pylist() == [3, 6, 9]
    import pickle

    assert pickle.loads(pickle.dumps(wire))(batch).column("x").to_pylist() == [3, 6, 9]


@_needs_cloudpickle
def test_the_strategy_now_admits_a_lambda_to_the_pool():
    assert strategy._process_capable(_op(lambda batch: batch, num_workers=4))


def test_an_unserializable_fn_is_declined_rather_than_dispatched():
    class Unserializable:
        def __call__(self, batch):
            return batch

        def __reduce__(self):
            raise TypeError("nope")

    assert processes.dispatchable(Unserializable()) is None
    assert not processes.is_picklable(Unserializable())


# --- the error budget means the same thing on both paths --------------------------------


def _always_fails(batch: pa.RecordBatch) -> pa.RecordBatch:
    raise ValueError("bad row")


def test_the_process_path_honours_max_errored_rows():
    """The same allowance, applied in the child, drops the rows instead of killing the query."""
    batches = [pa.record_batch({"x": pa.array([1, 2, 3, 4])})]
    results = processes.run_map_processes(
        _always_fails,
        batches,
        num_workers=2,
        batch_format="pyarrow",
        budget_key="tests.always_fails",
        max_errored_rows=8,
    )
    assert [b for shard in results for b in shard] == []


def test_the_process_path_still_fails_when_the_allowance_is_zero():
    batches = [pa.record_batch({"x": pa.array([1, 2])})]
    with pytest.raises(Exception, match="bad row"):
        processes.run_map_processes(_always_fails, batches, num_workers=1, batch_format="pyarrow")


def test_a_child_budget_is_shared_across_calls_in_that_process():
    a = processes._child_call(_always_fails, "pyarrow", "tests.shared", 4)
    b = processes._child_call(_always_fails, "pyarrow", "tests.shared", 4)
    from batcher.core.udf.call import shared_error_budget

    budget = shared_error_budget("tests.shared", 4)
    before = budget[0]
    a(pa.record_batch({"x": pa.array([1])}))
    b(pa.record_batch({"x": pa.array([2])}))
    assert budget[0] == before - 2  # one list, drawn down by both callables


# --- two local UDFs are two UDFs --------------------------------------------------------


def test_two_lambdas_in_one_scope_do_not_share_a_policy_key():
    first = _op(lambda batch: batch)
    second = _op(lambda batch: batch)
    assert strategy._fn_probe_key(first.fn) != strategy._fn_probe_key(second.fn)


def test_two_lambdas_in_one_scope_do_not_share_an_error_allowance():
    first = _op(lambda batch: batch, max_errored_rows=5)
    second = _op(lambda batch: batch, max_errored_rows=5)
    assert strategy.error_budget(first) is not strategy.error_budget(second)


def test_a_module_level_fn_keeps_a_line_independent_key():
    """The key is persisted across sessions, so it must not move when the file is edited."""
    assert strategy._fn_probe_key(_double) == f"{__name__}._double"


def test_a_row_adapter_carries_its_callbacks_line_not_just_its_name():
    from batcher.api.dataset.callbacks import _RowMap

    first = _RowMap(lambda row: row)
    second = _RowMap(lambda row: row)
    assert strategy._fn_probe_key(first) != strategy._fn_probe_key(second)


def test_the_serializability_answer_is_computed_once_per_callable():
    """The question is asked per stage invocation and answering it serializes the callable,
    so a UDF carrying a lookup table was dumping it every time to return a boolean."""
    calls: list[int] = []

    class Counting:
        def __call__(self, batch):
            return batch

        def __reduce__(self):
            calls.append(1)
            return (Counting, ())

    fn = Counting()
    assert processes.is_picklable(fn)
    first = len(calls)
    for _ in range(5):
        assert processes.is_picklable(fn)
    assert len(calls) == first
