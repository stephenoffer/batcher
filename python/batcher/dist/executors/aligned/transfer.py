"""Moving an aligned run's inputs and results between the driver and the fleet.

The broadcasts every unit joins go out once per node, compressed when large (`pack_held`,
`unpack_held`, decoded once per process); the units' results come back as they finish, with
the driver's budget checked while the rest still run (`gather_units`, `pull_units`).
"""

from __future__ import annotations

import dataclasses
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.logging import get_logger
from batcher.config.env import env_int

__all__ = [
    "PACK_BYTES",
    "RESULT_BYTES",
    "Packed",
    "gather_units",
    "pack_held",
    "pull_units",
    "read_unit",
    "result_budget",
    "trace_units",
    "unpack_held",
]

#: Bytes the units of one cut may return to the driver: groups, or rows a residual joins.
#: A ceiling: `result_budget` lowers it to what the driver's machine can actually hold.
RESULT_BYTES = env_int("BATCHER_ALIGNED_RESULT_BYTES", 12 << 30, floor=1 << 20)


def result_budget() -> int:
    """Bytes of unit results the driver may gather now: `RESULT_BYTES`, or less if it is short.

    `RESULT_BYTES` is a constant, and the driver is often the cluster's smallest machine: TPC-H
    q22 at SF1000 ran its driver on a 32 GB head node whose cgroup allows ~22 GB, beside an 8 GB
    object store, and the kernel killed it holding 18 GB. Half of what is available above twice
    the engine's headroom floor (`memory_headroom`) leaves room for the residual that runs over
    the gathered rows. A driver whose headroom cannot be read keeps the constant.
    """
    from batcher._internal.native import engine

    reading = engine().memory_headroom()
    if reading is None:
        return RESULT_BYTES
    available, floor = reading
    return min(RESULT_BYTES, max(1 << 20, (available - 2 * floor) // 2))


#: Broadcast bytes past which the held inputs travel compressed.
PACK_BYTES = 32 << 20


@dataclasses.dataclass(frozen=True)
class Packed:
    """One held broadcast as zstd Arrow IPC, named by its run so a process decodes it once.

    In `parts` independent streams, each a contiguous run of the batches, so both ends can
    compress and decompress them on several cores at once.
    """

    key: str
    parts: tuple[pa.Buffer, ...]


#: Most streams one held broadcast is cut into; each is (de)compressed on its own thread.
_PACK_PARTS = 8


def _write_stream(batches: list[pa.RecordBatch]) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(compression="zstd")
    with pa.ipc.new_stream(sink, batches[0].schema, options=options) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return sink.getvalue()


def _read_stream(buffer: pa.Buffer) -> list[pa.RecordBatch]:
    return pa.ipc.open_stream(buffer).read_all().to_batches()


def _contiguous(batches: list, parts: int) -> list[list]:
    """`batches` cut into at most `parts` contiguous runs of about equal bytes."""
    total = sum(b.nbytes for b in batches) or 1
    runs: list[list] = [[]]
    filled = 0
    for batch in batches:
        if runs[-1] and filled >= total * len(runs) / parts:
            runs.append([])
        runs[-1].append(batch)
        filled += batch.nbytes
    return runs


def pack_held(held: dict[int, object], key: str) -> dict[int, object]:
    """`held` as zstd-compressed Arrow IPC when it is large enough to be worth it.

    Every node fetches the broadcasts from the driver's object store, so the driver's link
    carries them once per node: TPC-H q14 at SF100 broadcasts `part` (20M rows, 725 MB) and
    its 100 units spent 11 s waiting on ~5.8 GB leaving one NIC, for 0.36 s of work each.
    Compressed it is 80 MB (sequential keys and a 150-value string column), 0.8 s to pack
    once and 0.4 s to unpack per task.

    Compressed on several threads, a contiguous run of batches each: one stream on one core
    took 8.8 s of TPC-H q9's 60 s at SF1000 on the driver, and 3-4 s of q8, q10 and q19.
    """
    # A recipe a node evaluates for itself (`local.LocalBroadcast`) travels as it is.
    rows = {sid: v for sid, v in held.items() if isinstance(v, list)}
    if sum(b.nbytes for batches in rows.values() for b in batches) < PACK_BYTES:
        return held
    packed: dict[int, object] = {sid: v for sid, v in held.items() if sid not in rows}
    jobs = [
        (sid, run) for sid, batches in rows.items() for run in _contiguous(batches, _PACK_PARTS)
    ]
    with ThreadPoolExecutor(max_workers=_PACK_PARTS) as pool:
        buffers = list(pool.map(lambda job: _write_stream(job[1]), jobs))
    for sid in rows:
        parts = tuple(buf for (owner, _run), buf in zip(jobs, buffers, strict=True) if owner == sid)
        packed[sid] = Packed(f"{key}:{sid}", parts)
    return packed


# Decoded broadcasts, per process: every call a node's actor serves for one run reads the
# same ones, and decoding them per call cost each call `part`'s 0.4 s on TPC-H q14 at SF100.
# Only the current run's are kept.
_DECODED: dict[str, list[pa.RecordBatch]] = {}
_DECODE_LOCK = threading.Lock()


def unpack_held(held: dict[int, object]) -> dict[int, object]:
    """`pack_held`'s inverse: every held input as record batches, decoded once per process."""
    packed = {sid: v for sid, v in held.items() if isinstance(v, Packed)}
    if not packed:
        return held
    with _DECODE_LOCK:
        live = {v.key for v in packed.values()}
        for stale in [k for k in _DECODED if k not in live]:
            del _DECODED[stale]
        todo = [v for v in packed.values() if v.key not in _DECODED]
        jobs = [(v.key, part) for v in todo for part in v.parts]
        with ThreadPoolExecutor(max_workers=max(1, min(_PACK_PARTS, len(jobs)))) as pool:
            decoded = list(pool.map(lambda job: _read_stream(job[1]), jobs))
        for v in todo:
            _DECODED[v.key] = [
                b
                for (owner, _part), bs in zip(jobs, decoded, strict=True)
                if owner == v.key
                for b in bs
            ]
        return {sid: _DECODED[v.key] if isinstance(v, Packed) else v for sid, v in held.items()}


def gather_units(calls: list[tuple], n_tasks: int, submit) -> list[tuple] | None:
    """Deal `calls` round-robin to `n_tasks` submissions and gather them in unit order."""
    n_tasks = max(1, n_tasks)
    submitted = time.time()
    pending = {}
    for t in range(n_tasks):
        units = list(range(t, len(calls), n_tasks))
        pending[submit(t, [calls[i] for i in units])] = (t, units)
    return _drain(calls, pending, lambda _worker, _ref: None, submitted)


#: Units per call when pulled: small enough that a slow node takes fewer, large enough that
#: each call's read of its next unit still overlaps the engine running its current one.
_PULL_CHUNK = 2

#: Seconds a call must have run before a copy of it is started, however quick the rest were.
_SPECULATE_AFTER_S = 0.2
#: Seconds between checks for a straggler to copy while nothing lands.
_IDLE_CHECK_S = 0.1


def pull_units(
    calls: list[tuple],
    workers: int,
    depth: int,
    submit: Callable[[int, list], object],
    keys: list[object] | None = None,
) -> list[tuple] | None:
    """Run `calls` on `workers`, `depth` calls in flight on each; results in unit order.

    Each worker is handed the next few units as it finishes some, rather than a fixed share
    up front: dealt round-robin, the slowest node set the wall time -- on TPC-H q17 at SF100
    the busiest of 16 streams finished at 4.2 s, where the mean one needed 1.5 s.

    Once every unit is handed out, a worker that frees up runs a copy of the call that has
    been running longest, if that is past twice the median call, and the first to finish is
    kept: units are deterministic, so either copy is the answer. One slow read otherwise sets
    the wall time -- at SF10, TPC-H q20's reads averaged 0.34 s and its slowest took 1.3 s.

    The copy that loses cannot be stopped on a threaded actor, so it holds one of that actor's
    call slots until it ends. `keys` names each worker across runs, and a worker still running
    an orphaned copy is given that many fewer calls: left out of the count, the next cut's
    calls queued behind them -- up to 1.1 s before a call began at SF10.
    """
    # Fewer units than call slots go one to a call, so every worker gets one.
    chunk = max(1, min(_PULL_CHUNK, len(calls) // max(1, workers * depth)))
    chunks = [list(range(i, min(i + chunk, len(calls)))) for i in range(0, len(calls), chunk)]
    keys = keys if keys is not None else list(range(workers))
    held_up = _orphans_on(keys)
    room = [max(1, depth - held_up[w]) for w in range(workers)]
    pending: dict[object, tuple[int, list[int]]] = {}
    started: dict[object, float] = {}
    twins: dict[object, object] = {}
    took: list[float] = []
    submitted, following = time.time(), 0

    def launch(worker: int, units: list[int]) -> object:
        ref = submit(worker, [calls[i] for i in units])
        pending[ref], started[ref] = (worker, units), time.monotonic()
        return ref

    def give(worker: int) -> None:
        nonlocal following
        if following < len(chunks):
            following += 1
            launch(worker, chunks[following - 1])
            return
        now = time.monotonic()
        limit = max(_SPECULATE_AFTER_S, 2.0 * statistics.median(took)) if took else None
        slow = [
            r
            for r, (w, _units) in pending.items()
            if r not in twins and w != worker and limit is not None and now - started[r] > limit
        ]
        if slow:
            original = min(slow, key=started.__getitem__)
            copy = launch(worker, pending[original][1])
            twins[original], twins[copy] = copy, original

    def landed(worker: int, ref: object) -> None:
        took.append(time.monotonic() - started.pop(ref))
        twin = twins.pop(ref, None)
        if twin is not None:  # its copy is no longer needed; a running actor call cannot stop
            twins.pop(twin, None)
            owner, _units = pending.pop(twin)
            started.pop(twin, None)
            with _ORPHAN_LOCK:
                _ORPHANS.append((keys[owner], twin))
        give(worker)

    def idle() -> None:
        # A straggler may outlast every other call, so nothing would land to trigger a copy.
        busy = [w for w, _units in pending.values()]
        for worker in range(workers):
            if busy.count(worker) < room[worker]:
                give(worker)

    for round_ in range(depth):
        for worker in range(workers):
            if round_ < room[worker]:
                give(worker)
    return _drain(calls, pending, landed, submitted, idle)


# Losing speculative copies still running, as (worker key, ref), across every run in this
# process: the cuts of one query, and the queries after it.
_ORPHANS: list[tuple[object, object]] = []
_ORPHAN_LOCK = threading.Lock()


def _orphans_on(keys: list[object]) -> list[int]:
    """How many still-running orphaned copies each worker in `keys` holds; drops finished ones."""
    import ray

    with _ORPHAN_LOCK:
        if _ORPHANS:
            refs = [ref for _key, ref in _ORPHANS]
            done, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
            finished = set(done)
            _ORPHANS[:] = [(k, r) for k, r in _ORPHANS if r not in finished]
        return [sum(1 for k, _r in _ORPHANS if k == key) for key in keys]


def _drain(
    calls: list[tuple],
    pending: dict[object, tuple[int, list[int]]],
    refill: Callable[[int, object], None],
    submitted: float,
    idle: Callable[[], None] | None = None,
) -> list[tuple] | None:
    """Land `pending` calls as they finish, refilling each worker that frees up.

    `refill(worker, ref)` runs after `ref` lands; it may add calls to `pending` or drop them.
    `idle()`, given, runs whenever `_IDLE_CHECK_S` passes with nothing landing.

    Gathered as they finish, so a result that would not fit on the driver is caught while
    the rest are still running and they are cancelled, not all landed first.
    """
    import ray

    out: list[tuple] = [()] * len(calls)
    landed = 0
    budget = result_budget()
    while pending:
        timeout = _IDLE_CHECK_S if idle is not None else None
        done, _ = ray.wait(list(pending), num_returns=1, timeout=timeout)
        if not done and idle is not None:
            idle()
        for ref in done:
            worker, units = pending.pop(ref)
            for i, result in zip(units, ray.get(ref), strict=True):
                out[i] = result
                landed += sum(b.nbytes for b in result[0])
            refill(worker, ref)
        if landed > budget:
            # Not `force=True`: these are actor calls, which Ray refuses to force-cancel
            # (`ValueError`), so the decline that was meant to fall back to another executor
            # failed TPC-H q9 and q10 at SF1000 instead.
            for ref in pending:
                ray.cancel(ref)
            return None
    # Seconds from submission to each call's first instruction: scheduling, a worker lease
    # and fetching the broadcasts -- the part of a small query that is neither read nor compute.
    return [(rows, m, (*t[:3], t[3] - submitted, *t[4:])) for rows, m, t in out]


def read_unit(descs: list, empties: dict[int, list]) -> tuple[list[list], int]:
    """Read one unit's aligned inputs, every source concurrently; returns them and their bytes.

    Each source's read is latency-bound on its own requests, so reading them one after
    another leaves the link half idle; they are independent, so they are read together.
    """
    from batcher.dist.executors.partition_io import read_partition_descriptor

    live = [i for i, d in enumerate(descs) if d is not None]
    inputs: list[list] = [[] for _ in descs]
    with ThreadPoolExecutor(max_workers=max(1, len(live))) as pool:
        read = pool.map(lambda i: read_partition_descriptor(descs[i]), live)
        for i, batches in zip(live, read, strict=True):
            # A unit whose row groups the range and the pushed predicate all prune reads
            # nothing, and the engine needs a schema to plan a join over it.
            inputs[i] = batches or empties[i]
    return inputs, sum(b.nbytes for batches in inputs for b in batches)


def trace_units(units, timings, hoist_s: float, gather_s: float, unit_cpus: int) -> None:
    """Log where an aligned run's time went: hoisting, and the units' read and compute."""
    reads = [t[0] for t in timings]  # time waiting on a read the prefetch did not hide
    execs = [t[1] for t in timings]
    gb = sum(t[2] for t in timings) / 1e9
    starts = [t[3] for t in timings if len(t) > 3]
    # A call's units share its start: what it spent, set-up included, bounds the wall time.
    busy: dict[float, float] = {}
    for t in timings:
        if len(t) > 4:
            busy[t[3]] = busy.get(t[3], t[4] + t[3]) + t[0] + t[1]
    get_logger("dist").info(
        "aligned: %d units x %d cpus; hoist %.2fs; units %.2fs wall; task start max %.2fs; "
        "last call done at %.2fs, set-up max %.2fs; "
        "per-unit read wait mean %.2fs max %.2fs, compute mean %.2fs max %.2fs; %.1f GB decoded",
        len(units),
        unit_cpus,
        hoist_s,
        gather_s,
        max(starts, default=0.0),
        max(busy.values(), default=0.0),
        max((t[4] for t in timings if len(t) > 4), default=0.0),
        sum(reads) / max(1, len(reads)),
        max(reads, default=0.0),
        sum(execs) / max(1, len(execs)),
        max(execs, default=0.0),
        gb,
    )
