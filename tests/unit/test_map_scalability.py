"""The map path's fan-out and dispatch stay bounded as the data and the call count grow.

Defects that all pass every gate while being wrong, because each is a *scheduling* decision
and scheduling never changes a result:

* the byte-derived partition count — the one that bounds a task's input memory — was
  clamped to the cluster's core count, so per-task memory grew linearly with the dataset;
* both compute weights (the fixed `_MAP_COMPUTE_WEIGHT` and the learned CPU factor) were
  multiplied into that same count rather than into the CPU a task reserves, which made every
  task smaller for no gain and let the two cancel each other;
* the per-batch dispatch pool was built and torn down inside every `map_batches` call,
  which a windowed / micro-batch stream makes once per window per stage;
* the streaming window was bounded in rows, which is not a memory bound at all when the rows
  are wide, and far below what memory allows when they are narrow.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pyarrow as pa
import pytest

from batcher.core.udf import apply as udf_apply
from batcher.dist.executors import map as mapmod
from batcher.plan.logical import MapBatches, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

_CORES = 128.0
_TARGET_BYTES = 256 * 1024 * 1024  # optimizer.target_bytes_per_task default


class _Split:
    def __init__(self, rows: int) -> None:
        self.rows = rows


class _Stats:
    def __init__(self, byte_size: int) -> None:
        self.byte_size = byte_size
        self.row_count = None


class _Source:
    """A splittable source that knows its own rows and bytes, as a footer/listing gives."""

    def __init__(self, rows: int, byte_size: int, n_splits: int) -> None:
        self._rows, self._bytes, self._n = rows, byte_size, n_splits

    def splits(self):
        return [_Split(self._rows // self._n) for _ in range(self._n)]

    def statistics(self):
        return _Stats(self._bytes)

    def identity(self) -> str:
        return "unit-source"


def _scan() -> Scan:
    return Scan(0, SchemaRef.from_arrow(pa.schema([("a", pa.int64())])))


def _map_plan() -> MapBatches:
    return MapBatches(_scan(), fn=lambda b: b)


def _partitions(source, plan) -> int:
    with patch.object(mapmod, "_cluster_cores", lambda: _CORES):
        return mapmod._adaptive_partition_count(source, plan, fallback=8)


def test_a_terabyte_scan_still_holds_one_task_to_the_byte_budget() -> None:
    """The memory bound is a bound, so the core-count clamp must not override it.

    A 1 TiB source needs ~4,096 partitions to keep each task's input at
    `target_bytes_per_task`. Clamping that to the 128 available cores gave 8 GiB per task —
    32x the budget — and the overshoot grows with the data, which is an OOM rather than a
    slow query.
    """
    total_bytes = 1 << 40
    src = _Source(rows=4_000_000_000, byte_size=total_bytes, n_splits=100_000)

    n = _partitions(src, _map_plan())

    assert n >= total_bytes // _TARGET_BYTES
    assert total_bytes / n <= _TARGET_BYTES


def test_a_wide_row_corpus_is_partitioned_by_bytes_not_by_rows() -> None:
    """Few, very wide rows: the row term is tiny and only the byte term is meaningful."""
    total_bytes = 800 * (1 << 30)  # 4 M rows of ~200 KB media
    src = _Source(rows=4_000_000, byte_size=total_bytes, n_splits=40_000)

    n = _partitions(src, _map_plan())

    assert total_bytes / n <= _TARGET_BYTES


def test_the_partition_count_never_exceeds_the_available_splits() -> None:
    """A task with no split to read is not a task — the byte term cannot invent one."""
    src = _Source(rows=4_000_000_000, byte_size=1 << 40, n_splits=16)

    assert _partitions(src, _map_plan()) <= 16


def test_a_small_source_still_runs_as_a_few_tasks() -> None:
    """The byte term must not inflate a fan-out the data does not justify."""
    src = _Source(rows=1_000_000, byte_size=40 << 20, n_splits=8)

    assert _partitions(src, _map_plan()) <= 8


def test_a_udf_does_not_shrink_the_tasks_it_runs_in() -> None:
    """`_MAP_COMPUTE_WEIGHT` sizes the CPU a task reserves, not how many tasks there are.

    Multiplying the *count* by it made every task correspondingly smaller — four times the
    dispatch, descriptor decoding, engine setup and worker acquisition for the same rows. It
    bought nothing, because a task's intra-task `num_workers` comes from the same CPU share
    the weight inflates, so the wider task runs the UDF just as many ways. Measured on an
    8-node cluster it was 1.4-2.0x slower at every UDF cost.
    """
    src = _Source(rows=64_000_000, byte_size=1_400_000_000, n_splits=256)

    assert _partitions(src, _map_plan()) == _partitions(src, _scan())


def test_a_udf_does_widen_the_cpu_it_reserves() -> None:
    """The other half: the weight has to land somewhere, and this is where."""
    partitions = [{"splits": [_Split(2_000_000)], "projection": None, "predicate": None}]

    with patch.object(mapmod, "_learned_weight_factor", lambda *a, **k: 1.0):
        for_map = mapmod._adaptive_task_cpus(partitions, _map_plan())
        for_scan = mapmod._adaptive_task_cpus(partitions, _scan())

    assert for_map[0] > for_scan[0]


def test_the_learned_packing_factor_does_not_shrink_cluster_parallelism() -> None:
    """`learned_cpu_weight_factor` reserves fewer *cores per task*; it must not run fewer
    tasks.

    Its own contract is "purely a packing decision — the rows a task processes are
    unchanged", and folding it into the partition count made that false. At its floor (0.25)
    it cancelled `_MAP_COMPUTE_WEIGHT` outright, so a map that had measured IO-bound ran a
    quarter of the tasks next time — which idles most of the cluster and makes the tasks
    *more* IO-bound, lowering the factor again.
    """
    src = _Source(rows=64_000_000, byte_size=1_400_000_000, n_splits=256)
    plan = _map_plan()

    with patch.object(mapmod, "_learned_weight_factor", lambda *a, **k: 1.0):
        unmeasured = _partitions(src, plan)
    with patch.object(mapmod, "_learned_weight_factor", lambda *a, **k: 0.25):
        measured_io_bound = _partitions(src, plan)

    assert measured_io_bound == unmeasured


def test_the_learned_packing_factor_still_shrinks_the_per_task_cpu_reservation() -> None:
    """The other half of the same contract: packing is exactly where it *should* apply."""
    # Big enough that both reservations clear the `_MIN_TASK_CPU` floor, or the two would
    # be equal for a reason that has nothing to do with the factor.
    partitions = [{"splits": [_Split(2_000_000)], "projection": None, "predicate": None}]
    plan = _map_plan()

    with patch.object(mapmod, "_learned_weight_factor", lambda *a, **k: 1.0):
        full = mapmod._adaptive_task_cpus(partitions, plan)
    with patch.object(mapmod, "_learned_weight_factor", lambda *a, **k: 0.25):
        packed = mapmod._adaptive_task_cpus(partitions, plan)

    assert packed[0] < full[0]


def test_the_engine_config_is_built_once_per_distinct_cpu_share() -> None:
    """Per distinct grant, not per task.

    The map path sizes each task's CPU grant from its own partition, so the config genuinely
    has to vary — but the shares repeat heavily (identical on a balanced scan), and
    `engine_config_json` re-serializes the active config and round-trips it through JSON on
    every call, on the one thread that also has to submit every task.
    """
    calls: list[float] = []

    def counting(share: float) -> str:
        calls.append(share)
        return f"cfg-{share}"

    with patch.object(mapmod, "engine_config_json", counting):
        cfg_for = mapmod._engine_config_cache()
        got = [cfg_for(s) for s in (1.0, 1.0, 1.0, 0.5, 1.0, 0.5)]

    assert calls == [1.0, 0.5]
    assert got == ["cfg-1.0"] * 3 + ["cfg-0.5", "cfg-1.0", "cfg-0.5"]


def test_sharing_the_plan_degrades_to_the_plan_itself_without_ray() -> None:
    """The object-store hand-off is an optimization, never a requirement: a stubbed or absent
    Ray must leave the task argument exactly what it was."""
    plan = _map_plan()

    with patch.dict("sys.modules", {"ray": None}):
        assert mapmod._shared_arg(plan) is plan


def test_a_dispatch_pool_is_reused_across_calls() -> None:
    """The pool a `map_batches` call runs its per-batch calls on outlives the call.

    `iter_batches` over a map chain calls into the stage once per *window*; building a
    `ThreadPoolExecutor` per call spawned 1,677 threads for a 16 M-row four-stage chain and
    made `__exit__` the whole profile.
    """
    udf_apply._IDLE_POOLS.clear()
    seen = []
    for _ in range(5):
        with udf_apply._leased_pool(3) as pool:
            seen.append(pool)

    assert len({id(p) for p in seen}) == 1


def test_concurrent_leases_get_their_own_pool() -> None:
    """Reuse is of *idle* pools only, so two stages running at once are as parallel as
    they were when each built its own."""
    udf_apply._IDLE_POOLS.clear()
    with udf_apply._leased_pool(2) as outer, udf_apply._leased_pool(2) as inner:
        assert outer is not inner


def test_a_leased_pool_still_bounds_concurrency_to_its_width() -> None:
    """The lease is exclusive, so `num_workers` remains the number of concurrent calls."""
    udf_apply._IDLE_POOLS.clear()
    live = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def work(_):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        gate.wait()  # only reachable if at least 2 run at once
        with lock:
            live -= 1

    with udf_apply._leased_pool(2) as pool:
        list(pool.map(work, range(8)))

    assert peak == 2


def test_parked_pools_are_bounded_across_widths() -> None:
    """Two-per-width is unbounded in the number of widths, so a total cap holds too.

    A long-lived process serving many differently-configured pipelines would otherwise park a
    pool per distinct `num_workers` and never release one — the same unbounded growth the
    reuse exists to remove, arrived at more slowly.
    """
    udf_apply._IDLE_POOLS.clear()
    for width in range(1, 12):
        with udf_apply._leased_pool(width):
            pass

    parked = sum(len(v) for v in udf_apply._IDLE_POOLS.values())
    assert parked <= udf_apply._MAX_IDLE_TOTAL
    udf_apply._IDLE_POOLS.clear()


def test_a_failed_lease_is_retired_rather_than_parked() -> None:
    """A pool whose lease raised may still hold queued calls, so it is not handed on."""
    udf_apply._IDLE_POOLS.clear()
    with pytest.raises(ValueError), udf_apply._leased_pool(2):
        raise ValueError("stage failed")

    assert not udf_apply._IDLE_POOLS.get(2)


# --- the streaming window is bounded in BYTES, not only in rows ------------------------


def _window_sizes(batches, target_rows):
    """Run `stream_windowed` over `batches`, returning (rows, bytes) per window and the rows."""
    from batcher.api.terminal.map_stream import stream_windowed
    from batcher.io.source import InMemorySource

    windows: list[tuple[int, int]] = []

    def run_window(buf):
        windows.append((sum(b.num_rows for b in buf), sum(b.nbytes for b in buf)))
        return buf

    out = list(stream_windowed(InMemorySource(batches), run_window, target_rows, None))
    return windows, out


def _batch(rows: int, width: int) -> pa.RecordBatch:
    """A batch of `rows` rows whose payload column is `width` bytes per row."""
    return pa.record_batch(
        {"i": pa.array(range(rows)), "blob": pa.array([b"x" * width] * rows, type=pa.binary())}
    )


def test_a_wide_row_window_flushes_on_bytes_not_rows() -> None:
    """A row count bounds nothing when rows are huge, which is exactly when it must.

    The same 245,760-row window is a few MB of narrow numerics and gigabytes of 8 KB blobs —
    and the multimodal scan is the shape whose consumer reached for `iter_batches` *because*
    the data does not fit. Measured before this bound existed: one window, 2,100 MB resident.
    """
    from batcher.api.terminal.map_stream import _WINDOW_BYTES

    wide = [_batch(2_000, 8 << 10) for _ in range(24)]  # ~16 MB per batch, ~390 MB total

    windows, out = _window_sizes(wide, target_rows=4_000_000)

    assert len(windows) > 1, "a 390 MB input must not arrive as a single window"
    # Overshoot is bounded by one source batch: the budget is checked after appending, so a
    # window can exceed it by whatever the batch that crossed the line contributed.
    assert max(b for _, b in windows) <= _WINDOW_BYTES + max(x.nbytes for x in wide)
    assert sum(r for r, _ in windows) == sum(b.num_rows for b in wide)
    assert sum(b.num_rows for b in out) == sum(b.num_rows for b in wide)


def test_a_narrow_window_still_fills_to_the_row_target() -> None:
    """The other end: narrow rows must not be cut into tiny windows.

    Each window pays a fixed cost (a plan walk, a re-chunk, a schema reconcile), and bounding
    a narrow stream at `workers x morsel` paid it tens of times more often than memory
    required — 1.9x on a four-stage chain over 8 M rows.
    """
    narrow = [_batch(10_000, 8) for _ in range(20)]  # tiny per row

    windows, _ = _window_sizes(narrow, target_rows=150_000)

    assert max(r for r, _ in windows) >= 150_000, "the byte bound must not bind on narrow rows"


def test_windowing_preserves_every_row_in_order() -> None:
    """Windowing is a scheduling choice, so it cannot change the rows or their order."""
    batches = [_batch(1_000, 4 << 10) for _ in range(12)]

    _, out = _window_sizes(batches, target_rows=4_000_000)

    expected = [v for b in batches for v in b.column("i").to_pylist()]
    assert [v for b in out for v in b.column("i").to_pylist()] == expected
