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
import pickle
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.errors import ExecutionError
from batcher._internal.hardware import available_cpu_count
from batcher._internal.mathx import ceil_div
from batcher.core.udf.isolation import (
    ResourceLimits,
    child_initializer,
    resolve_isolation,
    shard_directory,
)

__all__ = ["dispatchable", "run_map_processes", "shutdown_pool"]


def _pool_context():
    """Prefer a ``forkserver`` start method for the UDF process pool, else ``fork``.

    ``fork`` clones the parent whole-cloth — including any already-spun BLAS/OpenMP
    thread pool NumPy/torch created — so N forked workers each re-launch those threads
    and oversubscribe the cores (measured ~2x slower on a NumPy `fn`). ``forkserver``
    forks from a clean, minimal server process that never touched the math libs, so
    each worker starts single-threaded and the pool scales with cores. ``fork`` is the
    fallback where ``forkserver`` is unavailable (non-Unix).
    """
    from batcher._internal.hardware import process_start_method_context

    return process_start_method_context()


# A process pool is expensive to stand up (fork/forkserver N children), so a fresh one
# per `map_batches` call — or, worse, per streamed window — makes that startup dominate.
# Keep one warm pool for the process's lifetime and reuse it across every call, the way
# Ray Data reuses its actor pool. Lazily built, grown if a later call needs more workers.
_POOL: ProcessPoolExecutor | None = None
_POOL_SIZE = 0
_POOL_LOCK = None
# The isolation the live pool's children were started under. Part of the pool's identity,
# not just its configuration: the environment scrub happens once per child at startup, so
# a pool built under one setting cannot be reused under another. Without this, changing
# `udf_isolation` mid-process would silently keep serving children from the old regime.
_POOL_ISOLATION: tuple | None = None


def _persistent_pool(n: int, isolation: tuple) -> ProcessPoolExecutor:
    """The process's shared UDF pool, sized to at least `n` workers (grown lazily).

    One warm pool amortizes the fork/forkserver startup across every `map_batches` call
    and every streamed window, instead of paying it each time. Torn down at interpreter
    exit.

    Rebuilt when `isolation` changes, because each child applies it once at startup.

    Args:
        n: Minimum worker count.
        isolation: `(mode, allowed_env, limits)` from `resolve_isolation`.

    Returns:
        The shared pool.
    """
    global _POOL, _POOL_SIZE, _POOL_LOCK, _POOL_ISOLATION
    import atexit
    import threading

    if _POOL_LOCK is None:
        _POOL_LOCK = threading.Lock()
    mode, allowed, limits = isolation
    with _POOL_LOCK:
        if _POOL is None or n > _POOL_SIZE or isolation != _POOL_ISOLATION:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            else:
                atexit.register(shutdown_pool)
            # `initializer` runs once per child, so the scrub costs nothing per batch.
            # Under "none" no initializer is installed at all, leaving the old behaviour
            # byte-identical for an embedder who has opted out.
            kwargs = {}
            if mode != "none":
                kwargs = {
                    "initializer": child_initializer,
                    "initargs": (allowed, limits if mode == "strict" else ResourceLimits()),
                }
            _POOL = ProcessPoolExecutor(max_workers=n, mp_context=_pool_context(), **kwargs)
            _POOL_SIZE = n
            _POOL_ISOLATION = isolation
        return _POOL


def shutdown_pool() -> None:
    """Tear the shared UDF process pool down (idempotent; runs at interpreter exit)."""
    global _POOL, _POOL_SIZE, _POOL_ISOLATION
    if _POOL is not None:
        _POOL.shutdown(wait=False)
        _POOL = None
        _POOL_SIZE = 0
        _POOL_ISOLATION = None


def _apply_fn(fn: object, batch: pa.RecordBatch, fmt: str) -> object:
    """Apply `fn` to `batch`, converting to/from `fmt` when it is not ``pyarrow``."""
    if fmt == "pyarrow":
        return fn(batch)
    from batcher.interop.formats import result_to_arrowable, to_format

    return result_to_arrowable(fn(to_format(batch, fmt)), fmt)


def _call_shard(payload: tuple) -> list:
    """Map every batch of one shard, opening the memory-mapped file just once.

    The input never crosses the process boundary as pickled bytes: the worker maps the
    RAM-backed Arrow shard zero-copy and reads its batches in place. Processing the whole
    shard per task (rather than one batch per task) amortizes the map/open over all the
    shard's batches — the per-task open would otherwise dominate a moderate-input map.

    `budget_key` and `allowance` carry the stage's `max_errored_rows` policy into the child.
    Without them this path ignored the allowance entirely, so whether a corrupt row was
    dropped or killed the query depended on a *scheduling* decision the user never made —
    the strategy probe's threads-vs-processes verdict. The budget is per child process, which
    is exactly what the public contract already promises ("per worker").
    """
    fn, path, fmt, budget_key, allowance = payload
    call = _child_call(fn, fmt, budget_key, allowance)
    with pa.memory_map(path, "rb") as src:
        reader = pa.ipc.open_file(src)
        return [call(reader.get_batch(i)) for i in range(reader.num_record_batches)]


def _child_call(fn: object, fmt: str, budget_key: str | None, allowance: int):
    """The per-batch callable a worker runs: `fn` plus the dirty-row tolerance, if any.

    With no allowance this is the raw call and the child behaves exactly as before. With one,
    a failing batch is bisected to isolate the offending rows through the same
    `call._resilient_call` the thread path uses, so both paths drop the same rows and publish
    the same events. Retry/timeout is deliberately *not* applied here: see `resilience`.
    """
    if allowance <= 0:
        return lambda batch: _apply_fn(fn, batch, fmt)
    from batcher.core.udf.call import _resilient_call, shared_error_budget

    budget = shared_error_budget(budget_key or "process-pool", allowance)

    def _call(batch: pa.RecordBatch) -> object:
        return _resilient_call(lambda b: _apply_fn(fn, b, fmt), batch, budget, False)

    return _call


_SHM_COUNTER = 0


def _write_shard(path: str, batches: list[pa.RecordBatch]) -> None:
    """Write one shard, private to the running user from the moment it exists.

    The mode is set at `open` rather than by a following `chmod`: a chmod leaves a window
    in which the file is readable, and the whole point is that it never is.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with (
        os.fdopen(fd, "wb") as raw,
        pa.PythonFile(raw, mode="w") as sink,
        pa.ipc.new_file(sink, batches[0].schema) as writer,
    ):
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

    _SHM_COUNTER += 1
    # A private per-process directory rather than a flat name in world-writable
    # `/dev/shm`. These files ARE the query's data, and under the default umask they were
    # mode 0644 — any local user could read a running query's batches, which on a shared
    # box is a data leak that needs no query access at all.
    stem = os.path.join(shard_directory(), f"shard_{_SHM_COUNTER}")
    shard_size = ceil_div(len(batches), nshards)
    groups = [batches[g : g + shard_size] for g in range(0, len(batches), shard_size)]
    paths = [f"{stem}_{g}.arrow" for g in range(len(groups))]
    with ThreadPoolExecutor(max_workers=min(len(groups), available_cpu_count())) as pool:
        list(pool.map(_write_shard, paths, groups))
    return paths, shard_size


def run_map_processes(
    fn: object,
    batches: list[pa.RecordBatch],
    num_workers: int,
    batch_format: str,
    *,
    budget_key: str | None = None,
    max_errored_rows: int = 0,
) -> list[object]:
    """Apply `fn` to each batch across the warm process pool, preserving input order.

    `fn` is already the per-batch callable (a class factory is resolved by the caller).
    The input is written once to shared memory-mapped Arrow shards; each worker maps its
    shard zero-copy and reads its batches in place (no per-worker pickle of the input — the
    cost that otherwise loses a large string-heavy map to a plasma-backed engine). Only the
    `fn` and a shard path cross to the child; the results cross back over the pool. The pool
    is process-wide and reused, so there is no per-call startup cost.

    Args:
        fn: The per-batch callable, already built.
        batches: The stage's input batches, in order.
        num_workers: Upper bound on worker processes.
        batch_format: The object `fn` receives and returns.
        budget_key: Stable identity of the `fn`, so a child's error budget is shared across
            every shard and every call it runs.
        max_errored_rows: The stage's dirty-row allowance, applied per worker process.

    Returns:
        One result per input batch, in input order.
    """
    from batcher.config import active_config

    config = active_config()
    isolation = resolve_isolation(config)
    timeout = float(getattr(config.execution, "udf_timeout_s", 0.0) or 0.0)

    n_procs = max(1, min(num_workers, len(batches), available_cpu_count()))
    pool = _persistent_pool(n_procs, isolation)
    wire_fn = dispatchable(fn)
    if wire_fn is None:  # the strategy should have kept us on threads; be loud, not wrong
        raise ExecutionError(
            "a map_batches fn cannot be sent to a worker process (it is neither picklable "
            "nor cloudpickle-serializable).",
            hint="Install `cloudpickle`, or pass a module-level function or a class.",
        )
    paths, _size = _input_shards(batches, n_procs)
    try:
        # One task per shard (not per batch): the worker opens its mmap once and returns
        # all its results, so a moderate-input map isn't dominated by per-batch opens. Flat
        # in shard order == original batch order (shards are contiguous batch ranges).
        kwargs = {"timeout": timeout} if timeout > 0 else {}
        tasks = [(wire_fn, p, batch_format, budget_key, max_errored_rows) for p in paths]
        per_shard = pool.map(_call_shard, tasks, **kwargs)
        return [r for shard in per_shard for r in shard]
    except TimeoutError as exc:
        # A wedged UDF used to hang the query forever with no error and no signal. The
        # pool is torn down rather than reused: its children are still running the stuck
        # call, so handing them the next query's work would propagate the wedge.
        shutdown_pool()
        raise ExecutionError(
            f"a map_batches UDF did not finish within udf_timeout_s={timeout}s.",
            hint=(
                "Raise `execution.udf_timeout_s` if the UDF is legitimately slow, or set "
                "it to 0 to wait indefinitely."
            ),
        ) from exc
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)
        # Remove the per-process directory too, once it is empty; a long-lived process
        # would otherwise leave one behind per interpreter.
        with contextlib.suppress(OSError):
            os.rmdir(os.path.dirname(paths[0]))


#: Serialized-UDF size past which the payload is almost certainly captured *data* rather
#: than code. Ray warns at the same 10 MB and errors at 100 MB, and the guides are blunt
#: about what it means: "10MB+ function closure means a bug in your code" — a DataFrame or
#: a weight tensor caught in a closure and shipped with every dispatch.
_FAT_CLOSURE_BYTES = 10 << 20
_FAT_CLOSURE_WARNED = False


def warn_if_closure_is_fat(size: int) -> None:
    """Warn once when a UDF serializes to more than code plausibly should.

    The process path pickles the callable to each child, so whatever the callable *carries*
    rides along every time. Code is kilobytes; megabytes are data — a lookup table bound in
    with `functools.partial`, or a callable object holding one on `self`. The remedy is to
    load it inside a class UDF's `__init__` (once per worker) instead, which is the same
    shape the GPU model-reload warning already asks for.

    **What it cannot see.** `pickle` serializes a module-level function *by reference*, so a
    plain `def` that reads a large global pickles to a few bytes and is not flagged — Ray
    catches that case only because cloudpickle serializes globals by value, and the
    by-reference attempt here succeeds first. A closure over large data *is* flagged, but
    only where `cloudpickle` is installed: without it the closure cannot cross at all and
    the stage stays on threads, where nothing is shipped and there is nothing to warn about.

    The size comes from a pickle that had to happen anyway, so this measures nothing new.

    Args:
        size: The serialized size of the callable, in bytes.
    """
    global _FAT_CLOSURE_WARNED
    if size < _FAT_CLOSURE_BYTES or _FAT_CLOSURE_WARNED:
        return
    _FAT_CLOSURE_WARNED = True
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"this map_batches fn serializes to {size / (1 << 20):.0f} MB, which is data "
        f"captured from the enclosing scope rather than code — and it is shipped with "
        f"every dispatch. Load it inside a class UDF's __init__ so it is built once per "
        f"worker, and pass the class instead of the function.",
        PerformanceWarning,
        stacklevel=3,
    )


def _cloudpickle():
    """The `cloudpickle` module, or `None` when it is not installed.

    An optional dependency on purpose. Everything the engine itself ships across a process
    boundary is picklable by reference; cloudpickle only widens what a *user's* `fn` may be,
    so an install without it keeps every behaviour it had.
    """
    try:
        import cloudpickle

        return cloudpickle
    except ImportError:
        return None


def _load_by_value(blob: bytes) -> object:
    """Rebuild a by-value callable in the worker. Module-level so `pickle` can name it."""
    import cloudpickle

    return cloudpickle.loads(blob)


class _ByValueFn:
    """A callable that crosses to a worker **by value**, carrying its own serialized form.

    `pickle` sends a function by *reference* — module plus qualified name — which a lambda, a
    closure, or a class defined inside a function has no usable form of, so those simply
    refuse to pickle. That refusal is what used to bar the single most common spelling of a
    UDF from the process pool: ``ds.map_batches(lambda b: ...)`` stayed on threads and a
    GIL-bound body therefore ran on one core, however many were free. Ray, Dask, and Spark
    all reach for `cloudpickle` here, and so does this: the blob is built once on the driver
    and shipped with each dispatch, and the worker unpickles the real callable.

    Calling it locally runs the original object, so the driver-side probe and any thread
    fallback see the `fn` the user wrote, not a round-tripped copy of it.
    """

    __slots__ = ("_blob", "_fn")

    def __init__(self, fn: object, blob: bytes) -> None:
        self._fn = fn
        self._blob = blob

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._fn(*args, **kwargs)  # type: ignore[operator]

    def __reduce__(self) -> tuple:
        return (_load_by_value, (self._blob,))


def dispatchable(fn: object) -> object | None:
    """`fn` in the form a worker process can receive, or `None` when it cannot cross.

    Plain `pickle` is tried first, because a module-level function or a picklable callable
    object crosses by reference for a handful of bytes and needs nothing else. Only when that
    fails does the callable get serialized by value with `cloudpickle`, which is what admits
    lambdas and closures. Without `cloudpickle` installed the answer is `None` and the caller
    keeps the `fn` on threads, exactly as before.

    Args:
        fn: The callable a process worker would have to receive.

    Returns:
        The object to put in the dispatch payload, or None if the `fn` cannot be sent.
    """
    try:
        payload = pickle.dumps(fn)
    except Exception:
        payload = None
    if payload is not None:
        warn_if_closure_is_fat(len(payload))
        return fn
    cp = _cloudpickle()
    if cp is None:
        return None
    try:
        blob = cp.dumps(fn)
    except Exception:
        return None
    warn_if_closure_is_fat(len(blob))
    return _ByValueFn(fn, blob)


#: Answers already computed, keyed weakly by the callable so a `fn` that goes out of scope
#: takes its entry with it. The question is asked once per stage *invocation* — per partition,
#: per streamed window — and answering it means serializing the callable, so a UDF carrying a
#: lookup table was dumping megabytes each time purely to return a boolean.
_DISPATCHABLE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def is_picklable(obj: object) -> bool:
    """Whether `obj` can be sent to a worker process — and, for free, whether it is large.

    Lives here rather than beside the strategy that calls it because both halves are
    process-pool facts: only the process path serializes the callable, and the size
    `warn_if_closure_is_fat` judges comes out of that same dump.

    Args:
        obj: The callable a process worker would have to receive.

    Returns:
        True when it can cross the process boundary.
    """
    try:
        cached = _DISPATCHABLE.get(obj)
    except TypeError:  # not weak-referenceable (a builtin, say) — answer without caching
        return dispatchable(obj) is not None
    if cached is None:
        cached = dispatchable(obj) is not None
        with contextlib.suppress(TypeError):
            _DISPATCHABLE[obj] = cached
    return cached
