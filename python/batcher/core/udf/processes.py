"""The warm, shared process pool that runs CPU-bound `map_batches` UDFs off the GIL.

A pure-Python or NumPy `fn` that cannot release the GIL only scales across *processes*,
not threads. This module owns that pool and — critically — the way batches reach it:
the input is written once to RAM-backed (``/dev/shm``) memory-mapped Arrow shards that
every worker reads zero-copy, instead of pickling the whole input to each worker. That
shared read is what lets a large string-heavy map beat a plasma-backed engine instead of
bottlenecking on the driver serializing gigabytes per worker.

`bc-interp`'s `udf` module owns the *strategy* (when to use processes); this module owns
the *mechanism*. It takes an already-built callable, so it never imports `udf` back.
"""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count

__all__ = ["run_map_processes", "shutdown_pool"]


def _pool_context():
    """Prefer a ``forkserver`` start method for the UDF process pool, else ``fork``.

    ``fork`` clones the parent whole-cloth — including any already-spun BLAS/OpenMP
    thread pool NumPy/torch created — so N forked workers each re-launch those threads
    and oversubscribe the cores (measured ~2x slower on a NumPy `fn`). ``forkserver``
    forks from a clean, minimal server process that never touched the math libs, so
    each worker starts single-threaded and the pool scales with cores. ``fork`` is the
    fallback where ``forkserver`` is unavailable (non-Unix).
    """
    import multiprocessing as mp

    methods = mp.get_all_start_methods()
    return mp.get_context("forkserver" if "forkserver" in methods else "fork")


# A process pool is expensive to stand up (fork/forkserver N children), so a fresh one
# per `map_batches` call — or, worse, per streamed window — makes that startup dominate.
# Keep one warm pool for the process's lifetime and reuse it across every call, the way
# Ray Data reuses its actor pool. Lazily built, grown if a later call needs more workers.
_POOL: ProcessPoolExecutor | None = None
_POOL_SIZE = 0
_POOL_LOCK = None


def _persistent_pool(n: int) -> ProcessPoolExecutor:
    """The process's shared UDF pool, sized to at least `n` workers (grown lazily).

    One warm pool amortizes the fork/forkserver startup across every `map_batches` call
    and every streamed window, instead of paying it each time. Torn down at interpreter
    exit.
    """
    global _POOL, _POOL_SIZE, _POOL_LOCK
    import atexit
    import threading

    if _POOL_LOCK is None:
        _POOL_LOCK = threading.Lock()
    with _POOL_LOCK:
        if _POOL is None or n > _POOL_SIZE:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            else:
                atexit.register(shutdown_pool)
            _POOL = ProcessPoolExecutor(max_workers=n, mp_context=_pool_context())
            _POOL_SIZE = n
        return _POOL


def shutdown_pool() -> None:
    """Tear the shared UDF process pool down (idempotent; runs at interpreter exit)."""
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=False)
        _POOL = None
        _POOL_SIZE = 0


def _apply_fn(fn: object, batch: pa.RecordBatch, fmt: str) -> object:
    """Apply `fn` to `batch`, converting to/from `fmt` when it is not ``pyarrow``."""
    if fmt == "pyarrow":
        return fn(batch)
    from batcher.ml.batch_format import result_to_arrowable, to_format

    return result_to_arrowable(fn(to_format(batch, fmt)), fmt)


def _call_shard(payload: tuple) -> list:
    """Map every batch of one shard, opening the memory-mapped file just once.

    The input never crosses the process boundary as pickled bytes: the worker maps the
    RAM-backed Arrow shard zero-copy and reads its batches in place. Processing the whole
    shard per task (rather than one batch per task) amortizes the map/open over all the
    shard's batches — the per-task open would otherwise dominate a moderate-input map.
    """
    fn, path, fmt = payload
    with pa.memory_map(path, "rb") as src:
        reader = pa.ipc.open_file(src)
        return [_apply_fn(fn, reader.get_batch(i), fmt) for i in range(reader.num_record_batches)]


_SHM_COUNTER = 0


def _write_shard(path: str, batches: list[pa.RecordBatch]) -> None:
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, batches[0].schema) as writer:
        for b in batches:
            writer.write_batch(b)


def _input_shards(batches: list[pa.RecordBatch], nshards: int) -> tuple[list[str], int]:
    """Write `batches` to `nshards` RAM-backed (``/dev/shm``) random-access Arrow files.

    Sharded so the write parallelizes across a thread pool (Arrow IPC write runs GIL-free
    in C++) — a single file would serialize the whole input on one core. Workers read a
    shard zero-copy via `pa.memory_map`, so the input reaches every worker once through
    shared memory rather than a per-worker pickle. Returns the shard paths and the shard
    size (batches per shard); global batch `i` lives in shard ``i // shard_size`` at local
    index ``i % shard_size``. Falls back to the system temp dir when no tmpfs is present.
    """
    global _SHM_COUNTER
    import tempfile

    _SHM_COUNTER += 1
    root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    stem = os.path.join(root, f"bcudf_{os.getpid()}_{_SHM_COUNTER}")
    shard_size = -(-len(batches) // nshards)  # ceil-divide
    groups = [batches[g : g + shard_size] for g in range(0, len(batches), shard_size)]
    paths = [f"{stem}_{g}.arrow" for g in range(len(groups))]
    with ThreadPoolExecutor(max_workers=min(len(groups), available_cpu_count())) as pool:
        list(pool.map(_write_shard, paths, groups))
    return paths, shard_size


def run_map_processes(
    fn: object, batches: list[pa.RecordBatch], num_workers: int, batch_format: str
) -> list[object]:
    """Apply `fn` to each batch across the warm process pool, preserving input order.

    `fn` is already the per-batch callable (a class factory is resolved by the caller).
    The input is written once to shared memory-mapped Arrow shards; each worker maps its
    shard zero-copy and reads its batches in place (no per-worker pickle of the input — the
    cost that otherwise loses a large string-heavy map to a plasma-backed engine). Only the
    `fn` and a shard path cross to the child; the results cross back over the pool. The pool
    is process-wide and reused, so there is no per-call startup cost.
    """
    n_procs = max(1, min(num_workers, len(batches), available_cpu_count()))
    pool = _persistent_pool(n_procs)
    paths, _size = _input_shards(batches, n_procs)
    try:
        # One task per shard (not per batch): the worker opens its mmap once and returns
        # all its results, so a moderate-input map isn't dominated by per-batch opens. Flat
        # in shard order == original batch order (shards are contiguous batch ranges).
        per_shard = pool.map(_call_shard, [(fn, p, batch_format) for p in paths])
        return [r for shard in per_shard for r in shard]
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)
