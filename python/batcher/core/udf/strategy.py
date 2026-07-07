"""How a `map_batches` `fn` is run: threads vs processes, and the per-batch row count.

The `udf` module owns the *mechanism* (dispatch a per-batch call over threads/processes,
convert formats, isolate failures). This module owns the *policy* it consults:

- **threads vs processes** (`wants_processes`) — a vectorized `fn` (NumPy/Arrow/torch)
  runs fastest on threads with a coarse batch (zero serialization, input and output stay
  in shared memory); only a GIL-bound, cpu-heavy pure-Python `fn` is worth the process
  pool's shard-write + fork + result-pickle tax. The decision is *measured* on a small
  sample, not guessed from the row count.
- **batch size** (`thread_batch_target`) — the morsel (16,384) is tuned for the vectorized
  relational kernels, but a per-batch Python call pays a fixed overhead the morsel makes
  dominate; a light `fn` coarsens onto an amortization plateau, a heavy one keeps a
  per-worker split so every core stays busy.

Kept out of `udf` so that file stays the mechanism and this stays the (measured) policy.
It imports the callable builder from `udf` lazily, inside the probe, so there is no cycle.
"""

from __future__ import annotations

import pickle
import warnings

import pyarrow as pa

from batcher.plan.logical import MapBatches

__all__ = [
    "PROC_BATCHES_PER_WORKER",
    "PROC_MIN_BATCH_ROWS",
    "disable_processes",
    "map_strategy",
    "thread_batch_target",
    "wants_processes",
]

# When a process `fn` has no explicit `batch_size`, split the input into ~this many coarse
# batches per worker: enough for load balance, coarse enough that the per-batch pickle/IPC
# to the child is amortized (morsel-sized batches make that transfer dominate).
PROC_BATCHES_PER_WORKER = 3
# The smallest batch worth handing a process worker: below this the pickle/IPC per batch
# outweighs the work, so tinier batches are coalesced up to at least this many rows.
PROC_MIN_BATCH_ROWS = 65_536
# The smallest batch worth handing a *thread* worker for a light (GIL-releasing or cheap)
# `fn` with no explicit `batch_size`: a Python `map_batches` call pays a fixed per-call
# overhead (FFI + framework conversion + schema build) the morsel makes dominate. Measured,
# coarsening a NumPy `fn` from the morsel to this floor cut a 6 M-row map+reduce ~4x (108 ms
# -> ~27 ms); throughput is flat from here up, so this is the smallest batch on the plateau.
_THREAD_MIN_COARSE_ROWS = 131_072
# The coarsening ceiling: past this the per-call overhead is already amortized, and a
# smaller cap keeps more batches in flight (load balance) and a row-exploding / wide-row
# `fn`'s transient per-thread output bounded.
_THREAD_MAX_COARSE_ROWS = 262_144
# A `fn` whose measured per-row compute is below this is "light" — dominated by the fixed
# per-call overhead, so coarse batches win and leaving cores idle costs nothing. Above it,
# real per-row work makes filling every core matter, so keep the per-worker split. (NumPy
# `charge` measures ~7 ns/row; a pure-Python per-row transform is 10-100x that.)
_LIGHT_FN_ROW_SECONDS = 5e-8
# Thread/process cost probe (see `_prefer_processes`): time the `fn` on a sample of this
# many rows. Processes are chosen only when two concurrent thread calls fail to speed up by
# `_GIL_BOUND_MAX` (the `fn` holds the GIL) AND the estimated single-thread whole-job time
# exceeds `_PROC_WORTH_SECONDS` (cpu-heavy enough that multi-core beats the process overhead).
_PROBE_ROWS = 65_536
_PROBE_REPEATS = 3
_GIL_BOUND_MAX = 1.4
_PROC_WORTH_SECONDS = 0.5
# Below this row count the thread path keeps the morsel and never probes — coarsening can't
# change the batch count enough to matter, and a small query must not pay the probe latency.
_PROBE_MIN_ROWS = 262_144
# A CPU-bound `map_batches` big enough to amortize the process-pool startup can auto-run
# across processes; below this the thread path (no pool spawn) wins, so small queries keep
# their low fixed overhead.
_PROC_AUTO_MIN_ROWS = 1_000_000

# Set once if the process pool proves unusable this session (e.g. a non-import-safe
# entrypoint that forkserver/spawn cannot fork a child from). After that every stage stays
# on threads without re-attempting (and re-warning) a doomed pool per batch.
_processes_disabled = False
# The probe verdict per distinct `fn` — the measured trade-off is stable for a callable, so
# it is taken once and reused across queries (and streamed windows). Seeded from the hub on a
# miss (see `_learned_strategy`), so a recurring `fn` starts on the right pool across process
# restarts instead of re-probing every fresh session.
_PROC_PROBE_CACHE: dict[str, bool] = {}
# The probe's measured per-row compute (seconds) per `fn`, reused to size the thread batch:
# a light `fn` coarsens onto the plateau, a heavy one fills cores. Also hub-persisted so a
# recurring `fn` skips the re-probe.
_FN_ROW_SECONDS: dict[str, float] = {}
_REJECTED: set[str] = set()

# Hub namespace the measured per-`fn` policy (threads-vs-processes verdict + per-row seconds)
# persists under, keyed by the callable's `module.qualname`. Reading it seeds the in-process
# caches so a recurring UDF starts tuned across runs; a cold entry leaves the caches empty and
# the `fn` is probed exactly as before. Neither the verdict (threads vs processes) nor the
# batch size can change what a UDF computes, so a warm start is result-identical to a cold one.
_LEARN_NS = "udf_strategy"


def _learning_hub():
    """The process-wide MetadataHub, or `None` if unreachable — learned reads are best-effort."""
    try:
        from batcher.core.runtime import default_hub

        return default_hub()
    except Exception:  # pragma: no cover - learning must never break a query
        return None


def _learned_strategy(key: str) -> dict:
    """The persisted policy entry (``{"proc": bool, "row_secs": float}``) for a `fn` key, or
    ``{}`` when the hub is cold/unreachable. Best-effort — never raises into the probe."""
    hub = _learning_hub()
    if hub is None:
        return {}
    try:
        return hub.get_keyed_param(_LEARN_NS, key) or {}
    except Exception:  # pragma: no cover - learning must never break a query
        return {}


def _persist_strategy(key: str | None, **fields: object) -> None:
    """Merge `fields` into the persisted policy entry for a `fn` key (best-effort).

    Merges rather than overwrites so the verdict and the per-row seconds — measured by different
    probes — accumulate under one key without clobbering each other."""
    if key is None:
        return
    hub = _learning_hub()
    if hub is None:
        return
    try:
        entry = {**(hub.get_keyed_param(_LEARN_NS, key) or {}), **fields}
        hub.put_keyed_param(_LEARN_NS, key, entry)
    except Exception:  # pragma: no cover - learning must never break a query
        return


def disable_processes(exc: BaseException) -> None:
    """Disable the process path for the rest of the session, warning once."""
    global _processes_disabled
    if not _processes_disabled:
        _processes_disabled = True
        warnings.warn(
            f"map_batches process pool unavailable ({exc!r}); using threads for the rest "
            "of this session (for a CPU-bound pure-Python fn, run under an "
            "`if __name__ == '__main__':` guard so a worker process can start)",
            stacklevel=3,
        )


def wants_processes(op: MapBatches, total_rows: int, current: list[pa.RecordBatch]) -> bool:
    """Whether to run this `map_batches` across processes (vs threads).

    Processes when the user opted in (`op.multiprocessing`) or — adaptively — when a
    process-capable CPU `fn` is handed enough rows to amortize the pool-startup cost AND a
    measured cost comparison (`_prefer_processes`) shows their extra cores beat the
    result-pickle tax. A GIL-releasing vectorized `fn` (the common NumPy/Arrow/torch case)
    is routed to the cheaper thread path instead — threads keep input and output in shared
    memory, while the process path must ship results back. A GPU / class / non-pyarrow /
    unpicklable `fn` never qualifies (see `_process_capable`), nor does anything once the
    pool has proven unusable this session. The runtime still falls back to threads if a
    process actually fails, so this never drops a batch.
    """
    if op.num_workers <= 1 or _processes_disabled:
        return False
    if op.multiprocessing:
        return _process_safe(op)
    if total_rows < _PROC_AUTO_MIN_ROWS or not _process_capable(op):
        return False
    return _prefer_processes(op, total_rows, current)


def map_strategy(op: MapBatches, n_batches: int, use_processes: bool) -> str:
    """Pick how to run the per-batch calls: ``sequential``, ``threads``, or ``processes``.

    `use_processes` is the pre-computed intent (`wants_processes`); a single batch or a
    single worker collapses to sequential, everything else to threads.
    """
    if n_batches <= 1 or op.num_workers <= 1:
        return "sequential"
    return "processes" if use_processes else "threads"


def thread_batch_target(
    op: MapBatches, total_rows: int, num_workers: int, morsel: int, current: list[pa.RecordBatch]
) -> int:
    """Coarse per-batch row count for a threaded CPU `fn` with no explicit `batch_size`.

    The floor is one coarse batch per worker (`total / num_workers`, so every core is fed),
    capped at ``_THREAD_MAX_COARSE_ROWS`` (bounding a row-exploding / wide-row `fn`), never
    below the morsel. For a `fn` measured as *light* per row (a cheap vectorized transform
    whose cost is dominated by the fixed per-call overhead, not real work), the floor is
    lifted to the amortization plateau (`_THREAD_MIN_COARSE_ROWS`): a few coarse batches beat
    many morsel-sized ones because the per-call overhead — and GIL contention on any per-call
    Arrow conversion — falls, and the wasted work from fewer-than-core batches is nil when the
    `fn` is cheap. A `fn` measured as heavy (real per-row work) keeps the per-worker split so
    every core stays busy. The per-row cost is measured once on a sample and cached.
    """
    per_worker = -(-total_rows // max(1, num_workers))  # ceil-divide
    floor = morsel
    # Only worth probing (and coarsening) when the input is big enough for the batch count
    # to matter; a small query keeps the morsel and pays no probe latency.
    if total_rows > _PROBE_MIN_ROWS:
        row_secs = _fn_row_seconds(op, current)
        if row_secs is not None and row_secs < _LIGHT_FN_ROW_SECONDS:
            floor = _THREAD_MIN_COARSE_ROWS  # cheap per row -> coarsen onto the plateau
    return max(floor, min(_THREAD_MAX_COARSE_ROWS, max(morsel, per_worker)))


def _prefer_processes(op: MapBatches, total_rows: int, current: list[pa.RecordBatch]) -> bool:
    """Empirically decide threads vs processes for an auto (`multiprocessing` unset) `fn`.

    Batcher's process path reads its input zero-copy from shared memory but pays real,
    hard-to-model overhead the thread path never does: writing the input shards, forking the
    dispatch, and pickling every result back to the driver to unpickle + concatenate. That
    tax is only worth paying when a `fn` is BOTH GIL-bound (threads cannot spread it across
    cores) AND cpu-heavy enough that running it on a single thread would be genuinely slow —
    the pure-Python per-row transform Ray Data spreads across actors. A vectorized `fn`
    (NumPy/Arrow/torch) either releases the GIL or is cheap per row; both cases run fastest
    on threads with a coarse batch (no serialization at all), which is where the old
    row-count-only heuristic went wrong (it sent every vectorized `fn` down the slow path).

    On a small sample we measure the per-call compute and how much two concurrent threads
    speed it up (the GIL-release factor), then take the process path only when the `fn` is
    GIL-bound AND its estimated single-thread whole-job time clears a threshold where
    multi-core actually matters. Cached per callable; any probe failure falls back to the
    process path (conservative — a genuine GIL-bound `fn` still gets cores).
    """
    key = _fn_probe_key(op.fn)
    if key is not None and key in _PROC_PROBE_CACHE:
        return _PROC_PROBE_CACHE[key]
    # Warm start: a prior run's persisted verdict skips the probe entirely (and seeds the
    # in-process cache), so a recurring `fn` starts on the right pool across sessions.
    if key is not None:
        learned = _learned_strategy(key).get("proc")
        if isinstance(learned, bool):
            _PROC_PROBE_CACHE[key] = learned
            return learned
    try:
        verdict = _run_proc_probe(op, total_rows, current)
    except Exception:
        verdict = True  # probe couldn't run — keep the conservative process path
    if key is not None:
        _PROC_PROBE_CACHE[key] = verdict
        _persist_strategy(key, proc=verdict)
    return verdict


def _fn_row_seconds(op: MapBatches, current: list[pa.RecordBatch]) -> float | None:
    """Measured per-row compute (seconds) for `op.fn`, timed once on a sample and cached.

    Works for any callable (a closure/lambda too, unlike the process probe which needs a
    picklable `fn`), so the thread batch target can tell a light vectorized transform from a
    heavy per-row one. Returns None if it can't be measured (unkeyable / probe raised).
    """
    key = _fn_probe_key(op.fn)
    if key is not None and key in _FN_ROW_SECONDS:
        return _FN_ROW_SECONDS[key]
    # Warm start: reuse a prior run's measured per-row cost so the thread batch is sized right
    # from the first batch, without re-timing the `fn` this session.
    if key is not None:
        learned = _learned_strategy(key).get("row_secs")
        if isinstance(learned, (int, float)):
            _FN_ROW_SECONDS[key] = float(learned)
            return float(learned)
    try:
        secs = _measure_row_seconds(op, current)
    except Exception:
        secs = None
    if key is not None and secs is not None:
        _FN_ROW_SECONDS[key] = secs
        _persist_strategy(key, row_secs=secs)
    return secs


def _measure_row_seconds(op: MapBatches, current: list[pa.RecordBatch]) -> float | None:
    """Time `op.fn` on a small warmed sample; return seconds per row (None if no sample)."""
    import time

    sample = pa.Table.from_batches(current[:1]).slice(0, _PROBE_ROWS).to_batches()
    if not sample:
        return None
    probe = sample[0]
    rows = max(1, probe.num_rows)
    call = _probe_callable(op)
    call(probe)  # warm
    best = float("inf")
    for _ in range(_PROBE_REPEATS):
        t0 = time.perf_counter()
        call(probe)
        best = min(best, time.perf_counter() - t0)
    return best / rows


def _run_proc_probe(op: MapBatches, total_rows: int, current: list[pa.RecordBatch]) -> bool:
    """Measure the thread/process trade-off for `op.fn` on a small sample; True => processes."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    sample = pa.Table.from_batches(current[:1]).slice(0, _PROBE_ROWS).to_batches()
    if not sample:
        return True
    probe = sample[0]
    probe_rows = max(1, probe.num_rows)
    call = _probe_callable(op)
    call(probe)  # warm (imports, first-touch allocations, any lazy compile)

    def _best(runner) -> float:
        best = float("inf")
        for _ in range(_PROBE_REPEATS):
            t0 = time.perf_counter()
            runner()
            best = min(best, time.perf_counter() - t0)
        return best

    # Per-row compute (also cached for the thread batch target); derive the whole-call time.
    row_secs = _fn_row_seconds(op, current) or (_best(lambda: call(probe)) / probe_rows)
    t_call = row_secs * probe_rows
    pool = ThreadPoolExecutor(max_workers=2)

    def _two_concurrent() -> None:
        for f in (pool.submit(call, probe), pool.submit(call, probe)):
            f.result()

    try:
        par2 = _best(_two_concurrent)
    finally:
        pool.shutdown(wait=True)
    # Two calls concurrently vs one back-to-back: ~1x means the GIL serialized them (bound),
    # ~2x means it was released (they overlapped) and threads already scale the `fn`.
    thread_speedup = (2.0 * t_call / par2) if par2 > 0 else 1.0
    est_single_thread = t_call * (total_rows / probe_rows)
    return thread_speedup < _GIL_BOUND_MAX and est_single_thread > _PROC_WORTH_SECONDS


def _probe_callable(op: MapBatches):
    """Build the per-batch callable for the probe (Arrow in/out, or format-wrapped)."""
    from batcher.core.udf.execute import _formatted, build_udf_callable

    fn = build_udf_callable(op.fn)
    return fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)


def _fn_probe_key(fn: object) -> str | None:
    """A stable cache key for a `fn`'s probe verdict, or None if it can't be keyed."""
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    return f"{mod}.{qual}" if mod and qual else None


def _process_capable(op: MapBatches) -> bool:
    """Whether `op.fn` *can* run in a process pool (a quiet predicate, no warning).

    A factory/class would reload the model per child (and risk OOM); a GPU `fn` wants
    one CUDA context; a lambda/closure `fn` cannot be pickled to a spawned child. Any
    `batch_format` is fine — the numpy/pandas/torch conversion runs in the child from
    the picklable ``(fn, batch, fmt)`` payload, no closure required.
    """
    return not isinstance(op.fn, type) and op.num_gpus <= 0 and _is_picklable(op.fn)


def _process_safe(op: MapBatches) -> bool:
    """Whether `op.fn` can run in a process pool; warn-once and reject otherwise.

    The warning variant of `_process_capable`, used when the user *explicitly* asked for
    `multiprocessing=True` — so an ignored request is surfaced, not silently downgraded.
    """
    if isinstance(op.fn, type):
        return _reject("a factory/class fn would reload per process")
    if op.num_gpus > 0:
        return _reject("a GPU fn must keep a single process/CUDA context")
    if not _is_picklable(op.fn):
        return _reject("the fn is not picklable (a lambda or closure)")
    return True


def _reject(reason: str) -> bool:
    """Warn once per distinct reason that processes were declined, then return False."""
    if reason not in _REJECTED:
        _REJECTED.add(reason)
        warnings.warn(
            f"map_batches multiprocessing not used ({reason}); using threads",
            stacklevel=3,
        )
    return False


def _is_picklable(obj: object) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False
