"""Regression tests for the UDF execution path's memory bound and error budget.

Each test here pins a behavior that used to be wrong in a way every gate passed:

* `execute_with_udfs` called its result "streaming" while building a `list`, so peak memory
  was the whole output. `stream_with_udfs` is the genuinely bounded form, and the test proves
  the bound by counting how far the producer has run when the consumer takes its first batch.
* `max_errored_rows` was rebuilt per call, so every partition, window, and execution path got
  its own full allowance and the real bound scaled with parallelism.
* The per-row cost probe ran the user's `fn` up to four times on 65,536 rows before the query
  started, including a load-once model and a GPU forward.
* The autobatch path hardcoded its seed batch size and ran one dispatch slot, so the same
  model cold-started differently and lost its forward overlap depending on plan shape.

No GPU is needed: `num_gpus > 0` is a plan field, and the GPU-specific call wrappers
(`autocast_call`, the CUDA-OOM check) are no-ops on a host without a device.
"""

from __future__ import annotations

import threading
from typing import ClassVar

import pyarrow as pa
import pytest

from batcher.core.udf import apply as udf_apply
from batcher.core.udf import execute as udf_execute
from batcher.core.udf import strategy as udf_strategy
from batcher.core.udf import stream as udf_stream
from batcher.plan.logical import Limit, MapBatches, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([pa.field("x", pa.int64())])


class _CountingSource:
    """A source that records how many batches it has yielded, so a test can watch it run.

    Implements just the surface `core.udf` uses: `read` (the materializing path) and
    `iter_batches` (the streaming path). `identity` feeds the learned-readahead lookup.
    """

    bounded = True

    def __init__(self, n_batches: int, rows: int = 4) -> None:
        self._batches = [
            pa.record_batch({"x": list(range(i * rows, (i + 1) * rows))}, schema=_SCHEMA)
            for i in range(n_batches)
        ]
        self.produced = 0

    def identity(self) -> str:
        return "test:counting-source"

    def schema(self) -> pa.Schema:
        return _SCHEMA

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None):
        for batch in self._batches:
            self.produced += 1
            yield batch


def _gpu_chain(fn, **kwargs) -> MapBatches:
    """A `Scan -> map` plan that `stream_eligible` accepts (a lone GPU stage with a size)."""
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    kwargs.setdefault("batch_size", 4)
    return MapBatches(input=scan, fn=fn, num_gpus=1, **kwargs)


def _identity(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch


# ---------------------------------------------------------------------------
# 1. Bounded memory: the streaming form must not run the producer to completion
# ---------------------------------------------------------------------------


def test_stream_with_udfs_is_bounded_not_fully_materialized():
    """The consumer must see an early batch while most of the input is still unread.

    This is the memory claim itself, not a proxy for it: if the producer has already run to
    the end of a 200-batch source by the time the consumer takes batch 0, then every output
    batch is resident at once and the "streaming" path is bounded by the whole output. The
    ceiling below is the sum of the path's prefetch windows (source readahead, the per-stage
    queue, and the in-flight GPU window), all small constants.
    """
    n = 200
    src = _CountingSource(n)
    plan = _gpu_chain(_identity)

    it = udf_execute.stream_with_udfs(plan, [src])
    first = next(it)
    assert first.num_rows > 0
    assert src.produced < n, "the whole source was consumed before the first output batch"
    assert src.produced <= 32, f"look-ahead is not bounded: {src.produced} batches read"

    # ...and draining it still yields every row, in order.
    rest = list(it)
    got = pa.Table.from_batches([first, *rest]).column("x").to_pylist()
    assert got == list(range(n * 4))


def test_execute_with_udfs_still_returns_the_whole_result():
    """The listing entry point is unchanged — same rows, same order, for existing callers.

    Also the control for the bound above: this one *does* drain the source before returning
    anything, which is the property `stream_with_udfs` exists to avoid.
    """
    src = _CountingSource(8)
    out = udf_execute.execute_with_udfs(_gpu_chain(_identity), [src])
    assert pa.Table.from_batches(out).column("x").to_pylist() == list(range(32))
    assert src.produced == 8


def test_stream_with_udfs_falls_back_for_a_non_streaming_plan():
    """A plan the streaming path can't take still produces identical rows via the fallback."""
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    cpu_plan = MapBatches(input=scan, fn=_identity)  # no GPU stage -> not stream-eligible
    src = _CountingSource(5)
    got = list(udf_execute.stream_with_udfs(cpu_plan, [src]))
    assert pa.Table.from_batches(got).column("x").to_pylist() == list(range(20))


def test_reconcile_stream_widens_a_drifting_schema():
    """A UDF whose output gains a column stays concatenable without buffering everything."""
    batches = [
        pa.record_batch({"x": [1, 2]}),
        pa.record_batch({"x": [3, 4], "y": [5, 6]}),
    ]
    out = list(udf_stream.reconcile_stream(iter(batches)))
    # The later, wider batch is normalized to the union; the earlier one keeps its own schema
    # (it was already yielded), which is the documented weaker contract of the streaming form.
    assert out[1].schema.names == ["x", "y"]
    assert pa.Table.from_batches(out[1:]).column("y").to_pylist() == [5, 6]


def test_reconcile_stream_backfills_a_column_a_later_batch_drops():
    """After a widening, a batch that reverts to the narrow schema is backfilled to the union.

    The running schema only grows, so once a field appears every later batch is normalized to
    carry it (as a typed null when absent) — otherwise the stream would emit a batch narrower
    than the ones before it and break the downstream concat the widening exists to enable.
    """
    batches = [
        pa.record_batch({"x": [1]}),
        pa.record_batch({"x": [2], "y": [9]}),  # widens the running schema to {x, y}
        pa.record_batch({"x": [3]}),  # narrows again -> y must come back as a typed null
    ]
    out = list(udf_stream.reconcile_stream(iter(batches)))
    assert out[2].schema.names == ["x", "y"]
    assert out[2].column("y").to_pylist() == [None]


def test_reconcile_stream_of_an_empty_source_yields_nothing():
    assert list(udf_stream.reconcile_stream(iter([]))) == []


# ---------------------------------------------------------------------------
# An explicit batch_size must survive the parallel (num_workers > 1) CPU stage
# ---------------------------------------------------------------------------


def test_cpu_workers_respect_an_explicit_batch_size():
    """A parallel CPU stage must not re-slice an explicit `batch_size` to fill its pool.

    `_parallel_units` splits a morsel into `workers` even slices so a *zero-config* decode
    stage uses every spare core. When the user set a `batch_size`, doing that silently changes
    the batch boundaries the `fn` sees — violating the "explicit batch_size always wins" rule
    and handing a batch-boundary-sensitive `fn` the wrong-sized batches on the streaming path.
    """
    seen: list[int] = []

    def record(batch: pa.RecordBatch) -> pa.RecordBatch:
        seen.append(batch.num_rows)
        return batch

    op = MapBatches(
        input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=record, batch_size=100, num_workers=4
    )
    big = pa.record_batch({"x": list(range(200))}, schema=_SCHEMA)

    out = list(udf_stream._apply_udf_stream(iter([big]), op))

    # 200 rows at batch_size=100 -> exactly two 100-row calls, NOT four 50-row slices.
    assert sorted(seen) == [100, 100]
    # ...and the rows survive in order (pool.map preserves order).
    assert pa.Table.from_batches(out).column("x").to_pylist() == list(range(200))


# ---------------------------------------------------------------------------
# stream_eligible / linear_map_chain: the routing contract that picks the path
# ---------------------------------------------------------------------------


def _stage(**kwargs) -> MapBatches:
    return MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_identity, **kwargs)


def test_stream_eligible_lone_gpu_stage_needs_a_batch_size():
    assert udf_stream.stream_eligible([_stage(num_gpus=1, batch_size=4)]) is True
    # A zero-config lone GPU stage stays on the autobatch (hill-climbing) path instead.
    assert udf_stream.stream_eligible([_stage(num_gpus=1)]) is False


def test_stream_eligible_rejects_a_cpu_only_chain():
    assert udf_stream.stream_eligible([_stage()]) is False
    assert udf_stream.stream_eligible([_stage(), _stage()]) is False


def test_stream_eligible_accepts_a_multi_stage_chain_with_a_gpu():
    assert udf_stream.stream_eligible([_stage(), _stage(num_gpus=1)]) is True


def test_stream_eligible_rejects_a_multiprocessing_stage():
    # A multiprocessing stage runs across processes -> the materializing path regardless.
    assert udf_stream.stream_eligible([_stage(multiprocessing=True), _stage(num_gpus=1)]) is False


def test_linear_map_chain_returns_stages_bottom_up():
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    inner = MapBatches(input=scan, fn=_identity)
    outer = MapBatches(input=inner, fn=_identity)
    got = udf_stream.linear_map_chain(outer)
    assert got is not None
    root, stages = got
    assert root is scan
    assert stages == [inner, outer]  # bottom-up, so stage order matches execution order


def test_linear_map_chain_rejects_a_plan_with_no_map():
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    assert udf_stream.linear_map_chain(scan) is None


def test_linear_map_chain_rejects_a_non_scan_root():
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    plan = MapBatches(input=Limit(input=scan, n=3), fn=_identity)  # a relational node breaks it
    assert udf_stream.linear_map_chain(plan) is None


# ---------------------------------------------------------------------------
# 2. The error budget is one allowance per worker, not one per call
# ---------------------------------------------------------------------------


def _always_raises(batch: pa.RecordBatch) -> pa.RecordBatch:
    raise ValueError("bad row")


def test_error_budget_is_shared_across_calls_not_reset_per_call():
    """A second `_apply_udf` call draws down the SAME allowance the first one spent.

    Before this, `budget = [op.max_errored_rows]` was rebuilt per invocation, so N partitions
    meant N x `max_errored_rows` dropped rows and the documented bound was meaningless.
    """
    op = MapBatches(
        input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_always_raises, max_errored_rows=4
    )
    batch = pa.record_batch({"x": [1, 2, 3, 4]}, schema=_SCHEMA)

    assert udf_apply.apply_udf([batch], op) == []  # 4 rows dropped, budget now 0
    with pytest.raises(ValueError):
        udf_apply.apply_udf([batch], op)  # a fresh budget would have swallowed these too


def _also_raises(batch: pa.RecordBatch) -> pa.RecordBatch:
    raise ValueError("bad row")


def test_error_budget_is_the_same_object_on_both_execution_paths():
    """The materializing path and the streaming path must not each hold their own allowance.

    A query that routes through both (a plan shape change between stages, or a streaming
    query calling in once per window) used to get two full budgets for one operator.
    """
    op = MapBatches(
        input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_also_raises, max_errored_rows=7
    )
    assert udf_strategy.error_budget(op) is udf_strategy.error_budget(op)


def test_error_budget_separates_distinct_functions():
    """Two different UDFs must not share one allowance."""
    scan = Scan(0, SchemaRef.from_arrow(_SCHEMA))
    a = MapBatches(input=scan, fn=_always_raises, max_errored_rows=3)
    b = MapBatches(input=scan, fn=_identity, max_errored_rows=3)
    assert udf_strategy.error_budget(a) is not udf_strategy.error_budget(b)


# ---------------------------------------------------------------------------
# 3. The per-row cost probe must not run an expensive or side-effecting fn
# ---------------------------------------------------------------------------


class _ModelUDF:
    """A load-once class UDF — the shape whose probe used to run inference four times."""

    calls = 0

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        type(self).calls += 1
        return batch


def test_probe_never_calls_a_class_model_udf():
    """A factory/class `fn` is a loaded model; timing it costs real forward passes."""
    _ModelUDF.calls = 0
    op = MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_ModelUDF, num_workers=4)
    batch = pa.record_batch({"x": list(range(1000))}, schema=_SCHEMA)

    assert udf_strategy._fn_row_seconds(op, [batch]) is None
    assert _ModelUDF.calls == 0, "the probe instantiated and ran the model"


def _gpu_probe_fn(batch: pa.RecordBatch) -> pa.RecordBatch:
    _gpu_probe_fn.calls += 1
    return batch


_gpu_probe_fn.calls = 0


def test_probe_never_calls_a_gpu_fn():
    """A GPU `fn`'s probe is a device forward pass on a device the query is about to need."""
    _gpu_probe_fn.calls = 0
    op = MapBatches(
        input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_gpu_probe_fn, num_gpus=1, num_workers=4
    )
    batch = pa.record_batch({"x": list(range(1000))}, schema=_SCHEMA)

    assert udf_strategy._fn_row_seconds(op, [batch]) is None
    assert _gpu_probe_fn.calls == 0


def _slow_fn(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Stands in for a `fn` with a real per-call cost (a paid API, a heavy transform)."""
    _slow_fn.calls += 1
    threading.Event().wait(0.06)  # > _PROBE_TIME_BUDGET_SECONDS
    return batch


_slow_fn.calls = 0


def test_probe_of_an_expensive_fn_costs_one_call():
    """A `fn` already slow on the sample is decisively heavy; repeating the timing is waste."""
    _slow_fn.calls = 0
    udf_strategy._FN_ROW_SECONDS.pop("tests.unit.test_udf_streaming_and_budget._slow_fn", None)
    op = MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_slow_fn, num_workers=4)
    batch = pa.record_batch({"x": list(range(1000))}, schema=_SCHEMA)

    secs = udf_strategy._measure_row_seconds(op, [batch])
    assert secs is not None and secs > 0
    assert _slow_fn.calls == 1, f"probe ran the expensive fn {_slow_fn.calls} times"


# ---------------------------------------------------------------------------
# 4/5. The autobatch path shares the streaming path's learned seed and overlap
# ---------------------------------------------------------------------------


class _RecordingPool:
    """Stands in for `ml.inference.InferencePool`, capturing how it was configured."""

    last: ClassVar[dict] = {}

    def __init__(self, factory, **kwargs) -> None:
        type(self).last = dict(kwargs)
        self._worker = factory()

    def run(self, batches):
        for batch in batches:
            yield self._worker(batch)


def _autobatch_fn(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch


def test_autobatch_seeds_from_the_learned_gpu_batch_size(monkeypatch):
    """The materializing GPU path must use the same learned seed the streaming path does.

    It hardcoded `target_batch_rows=256`, so the same model cold-started two different ways
    depending on which path the plan shape routed it to, discarding what the last run learned.
    """
    import batcher.ml.inference as inference

    monkeypatch.setattr(inference, "InferencePool", _RecordingPool)
    monkeypatch.setattr(udf_stream, "_read_ema", lambda ns, key: 64.0)

    op = MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_autobatch_fn, num_gpus=1)
    batch = pa.record_batch({"x": [1, 2, 3, 4]}, schema=_SCHEMA)
    out = udf_apply._apply_udf_autobatch(op, [batch])

    assert pa.Table.from_batches(out).column("x").to_pylist() == [1, 2, 3, 4]
    assert _RecordingPool.last["target_batch_rows"] == udf_stream._learned_gpu_cap(op) == 64


def test_autobatch_gets_the_same_forward_overlap_as_the_streaming_path(monkeypatch):
    """A solo GPU stage resolves to `num_workers == 1`, which left the pool with one slot."""
    import batcher.ml.inference as inference

    monkeypatch.setattr(inference, "InferencePool", _RecordingPool)

    op = MapBatches(input=Scan(0, SchemaRef.from_arrow(_SCHEMA)), fn=_autobatch_fn, num_gpus=1)
    assert op.num_workers <= 1  # the condition that used to disable overlap entirely
    udf_apply._apply_udf_autobatch(op, [pa.record_batch({"x": [1]}, schema=_SCHEMA)])

    assert _RecordingPool.last["num_workers"] == udf_stream._GPU_SOLO_PIPELINE_DEPTH > 1
