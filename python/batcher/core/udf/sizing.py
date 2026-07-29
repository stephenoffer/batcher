"""What the streaming UDF path learned last run, folded back into this run's sizing.

Three sizing questions the stage scheduler asks have no good cold answer: how many rows a
model can hold on the device, how wide a morsel a CPU stage should take, and how far ahead
to read a source. Guessing costs a GPU OOM or an idle device; measuring costs a run. So each
is measured once and folded into a per-signature EMA in the MetadataHub, and the next run of
the same shape starts from what the last one proved.

**None of this can change a result.** A batch or chunk size only shards rows, and a prefetch
depth only reorders when a chunk is read, so a warm start is byte-identical to a cold one.
That is what makes learning safe to apply unconditionally, and it is why this module holds
sizing only — never a decision about *what* to compute.

Split from `stream` on the seam its author had already marked: everything here is a pure
function of the hub plus one observation, with no morsel scheduling and no UDF call.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.metadata.hardware_scope import scoped
from batcher.plan.logical import MapBatches

__all__ = [
    "cpu_batch_rows",
    "fold_ema",
    "gpu_batch_rows",
    "learned_gpu_cap",
    "learned_read_depth",
    "stage_sig",
    "timed_source",
    "warn_if_row_is_unsplittable",
]

# --- cold-start sizing defaults ------------------------------------------------------
# What the learned refinements below start from, kept beside them so a default and the EMA
# that narrows it cannot drift apart across two modules.

# Bounded look-ahead between pipelined map stages: a stage may run this many morsels ahead
# of its consumer (so a CPU stage overlaps the GPU stage draining it) while keeping resident
# memory to ~`depth` morsels per stage. Env-overridable.
_STREAM_PREFETCH_DEPTH = max(0, int(os.environ.get("BATCHER_STREAM_PREFETCH_DEPTH", "2")))
# The deepest source-read look-ahead the learned readahead may request (a slow source hides
# more of its latency behind compute); bounds resident memory to ~this many morsels.
_STREAM_MAX_PREFETCH_DEPTH = max(
    _STREAM_PREFETCH_DEPTH, int(os.environ.get("BATCHER_STREAM_MAX_PREFETCH_DEPTH", "8"))
)
# Adaptive GPU-inference batch when a GPU stage has no explicit `batch_size` (the truly
# zero-config `ds.map_batches(Model, num_gpus=1)` call). `_GPU_STREAM_BATCH_ROWS` is the row
# cap (large enough to fill the device; the guides' image range is 32-128, 256 suits most
# vision/embedding models); `_GPU_STREAM_BATCH_BYTES` is a per-batch input-byte budget so the
# row count SHRINKS on wide rows (a decoded frame, a float embedding tensor) that would
# otherwise OOM the GPU at the row cap, and stays at the cap for narrow rows. Floored so the
# batch always fills the SMs. An explicit `batch_size` always wins; env-overridable.
_GPU_STREAM_BATCH_ROWS = max(1, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_ROWS", "256")))
_GPU_STREAM_BATCH_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_BYTES", str(64 << 20)))
)
_GPU_STREAM_BATCH_MIN = max(1, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_MIN", "8")))
# Per-batch input-byte budget for a CPU (decode/preprocess) stage with no explicit
# `batch_size`: like the GPU budget, this SHRINKS the chunk below the morsel when a stage's
# rows are huge so a transient per-thread output stays bounded, and keeps the full morsel for
# narrow rows. Result-invariant -- the chunk only shards. Env-overridable.
_CPU_STREAM_BATCH_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_CPU_STREAM_BATCH_BYTES", str(128 << 20)))
)


def gpu_batch_rows(batch: pa.RecordBatch, row_cap: int = _GPU_STREAM_BATCH_ROWS) -> int:
    """Adaptive GPU sub-batch row count for a morsel, from its per-row byte width.

    ``min(row_cap, byte_budget / per_row_bytes)`` floored at `_GPU_STREAM_BATCH_MIN`: narrow
    rows batch up to the row cap (fill the device); wide rows (large images/tensors) batch
    fewer rows to stay under the VRAM budget -- data-width-adaptive, so the same zero-config
    call is safe for a 150 KB image and a 3 MB frame alike (OOM-halving covers the rest). The
    `row_cap` is the model's learned VRAM-safe size when known (see `learned_gpu_cap`), else
    the config default.

    Args:
        batch: The morsel about to be sub-batched.
        row_cap: The upper bound on rows per sub-batch.

    Returns:
        Rows per GPU sub-batch for this morsel.
    """
    if batch.num_rows <= 0:
        return row_cap
    per_row = max(1, batch.nbytes // batch.num_rows)
    warn_if_row_is_unsplittable(per_row)
    by_bytes = _GPU_STREAM_BATCH_BYTES // per_row
    return max(_GPU_STREAM_BATCH_MIN, min(row_cap, by_bytes))


# Hub namespaces for the streaming path's learned sizing, keyed by a stable per-stage / per-source
# signature. None of these can change a UDF's result — a batch/chunk size only shards rows, a
# prefetch depth only reorders when a chunk is read — so a warm start is byte-identical to cold.
_GPU_BATCH_NS = "udf_gpu_batch"  # learned VRAM-safe GPU batch rows per model signature
_SCAN_TPUT_NS = "udf_scan_tput"  # learned source read throughput (rows/sec) per source identity


def _stream_hub():
    """The process-wide MetadataHub, or `None` if unreachable — learned reads are best-effort."""
    try:
        from batcher.core.runtime import default_hub

        return default_hub()
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("core", "resolve metadata hub", exc)
        return None


def stage_sig(op: MapBatches) -> str | None:
    """A stable per-stage signature for `op` (its UDF's ``module.qualname``), or `None`."""
    fn = op.fn
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    return f"{mod}.{qual}" if mod and qual else None


def fold_ema(namespace: str, key: str | None, value: float) -> None:
    """Fold one observation into a per-signature EMA bucket ``{ema, n}`` in the hub. Best-effort.

    Stored under a hardware-scoped namespace, because every value that reaches here is a size
    chosen against the machine — VRAM-safe GPU batch rows, a source's read throughput — and
    none of them transfers to different hardware. The scoping is applied here and in
    `_read_ema` rather than at the call sites, so a read and a write cannot disagree about
    which namespace they mean.

    (A compact local copy of the dist learner's fold — `core` cannot import the `dist` layer.)"""
    if key is None or value != value or value <= 0.0:
        return
    hub = _stream_hub()
    if hub is None:
        return
    try:
        from batcher.config import active_config

        s = hub.get_keyed_param(scoped(namespace), key) or {}
        a = float(active_config().optimizer.learning_smoothing_alpha)
        prior = s.get("ema")
        ema = value if prior is None else a * value + (1.0 - a) * float(prior)
        hub.put_keyed_param(scoped(namespace), key, {"ema": ema, "n": int(s.get("n", 0)) + 1})
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("core", "fold learned ema", exc)
        return


def _read_ema(namespace: str, key: str | None) -> float | None:
    """The learned EMA for a signature (best-effort), or `None` when cold/unreachable."""
    if key is None:
        return None
    hub = _stream_hub()
    if hub is None:
        return None
    try:
        s = hub.get_keyed_param(scoped(namespace), key) or {}
    except Exception as exc:  # pragma: no cover
        note_suppressed("core", "read learned ema", exc)
        return None
    return float(s["ema"]) if "ema" in s else None


def learned_gpu_cap(op: MapBatches) -> int:
    """The GPU batch-row cap for a model, seeded from its learned VRAM-safe size when known.

    A prior run's settled adaptive batch size (persisted per model signature) caps the byte-budget
    sizing so the next run starts at the learned safe size instead of rediscovering it from the
    default row cap — the automatic form of "we already found this model fits N rows". Never above
    the config cap; a cold model keeps the cap. OOM-halving remains the in-run safety net."""
    learned = _read_ema(_GPU_BATCH_NS, stage_sig(op))
    cap = _GPU_STREAM_BATCH_ROWS
    if learned is not None and learned >= 1.0:
        cap = min(cap, int(learned))
    return max(_GPU_STREAM_BATCH_MIN, cap)


def cpu_batch_rows(batch: pa.RecordBatch, morsel: int) -> int:
    """Byte-adaptive chunk row count for a CPU stage with no explicit `batch_size`.

    ``min(morsel, byte_budget / per_row_bytes)`` floored at 1: narrow rows keep the full morsel;
    wide (post-decode multimodal) rows chunk fewer so a per-thread output stays bounded. The chunk
    only shards the morsel, so the concatenated output is identical to the plain morsel path."""
    if batch.num_rows <= 0:
        return morsel
    per_row = max(1, batch.nbytes // batch.num_rows)
    warn_if_row_is_unsplittable(per_row)
    by_bytes = _CPU_STREAM_BATCH_BYTES // per_row
    return max(1, min(morsel, by_bytes))


def learned_read_depth(source) -> int:
    """The source-read prefetch depth, deepened for a source measured as slow to read.

    Reads the source's learned throughput (rows/sec, persisted per identity): a slow source (remote
    object storage, a throttled connector) gets a deeper look-ahead so more chunks overlap compute,
    while a fast local source keeps the base depth (deeper prefetch would only add resident memory).
    Clamped to ``[_STREAM_PREFETCH_DEPTH, _STREAM_MAX_PREFETCH_DEPTH]``. Prefetch only reorders when
    a chunk is read, never which rows it holds, so the result is identical at any depth."""
    try:
        ident = source.identity()
    except Exception as exc:  # pragma: no cover
        note_suppressed("core", "read source identity", exc)
        return _STREAM_PREFETCH_DEPTH
    tput = _read_ema(_SCAN_TPUT_NS, ident)
    if tput is None:
        return _STREAM_PREFETCH_DEPTH
    import math

    fast = 5_000_000.0  # rows/sec: a fast local scan needs no extra readahead
    if tput >= fast:
        return _STREAM_PREFETCH_DEPTH
    extra = int(math.log2(fast / max(tput, 1.0)))
    deep = _STREAM_PREFETCH_DEPTH + extra
    return max(_STREAM_PREFETCH_DEPTH, min(_STREAM_MAX_PREFETCH_DEPTH, deep))


def timed_source(source, gen: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Yield the source's morsels, recording its measured read throughput (rows/sec) on exhaustion.

    Times only the source iteration (rows / elapsed) and folds it into the learned readahead
    signal, so the next run can deepen the prefetch for a slow source. Timing is a driver-side
    counter — it touches no row and cannot change what is yielded."""

    ident = None
    with contextlib.suppress(Exception):
        ident = source.identity()
    rows = 0
    t0 = time.perf_counter()
    try:
        for batch in gen:
            rows += batch.num_rows
            yield batch
    finally:
        elapsed = time.perf_counter() - t0
        if ident is not None and rows > 0 and elapsed > 0.0:
            fold_ema(_SCAN_TPUT_NS, ident, rows / elapsed)


#: Per-row width past which byte-adaptive batching has nothing left to give: the chunk is
#: already one row, and **a row cannot be split**. The field guides state this one
#: unconditionally — "a single row larger than available task memory causes an unrecoverable
#: OOM regardless of any other tuning... a hard architectural constraint, not a guideline" —
#: and put the safe ceiling at ~10 MB. This sits higher on purpose: at 10 MB the byte budget
#: still yields a chunk of ~12 rows, so batching is working and the advice would be false.
#: 64 MB is where the chunk reaches its floor (2 rows on CPU, 1 on GPU) — the last point at
#: which anything can be said before the allocation that fails.
_UNSPLITTABLE_ROW_BYTES = 64 << 20
#: Warned once per process. This runs per morsel on the hot path, and a corpus of huge rows
#: would otherwise emit the same warning thousands of times.
_ROW_WIDTH_WARNED = False


def warn_if_row_is_unsplittable(per_row_bytes: int) -> None:
    """Warn once when a single row is so wide that batching can no longer bound memory.

    Every other sizing lever here shrinks the chunk until the working set fits. That works
    down to one row and then stops: rows are atomic. Past this width the engine has spent
    its last adaptive move, so the remaining fix belongs to the data — split the blob out of
    the row, keep a reference, or decode later — and the user can only make it if told.

    Args:
        per_row_bytes: The measured average width of a row in the batch.
    """
    global _ROW_WIDTH_WARNED
    if per_row_bytes < _UNSPLITTABLE_ROW_BYTES or _ROW_WIDTH_WARNED:
        return
    _ROW_WIDTH_WARNED = True
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"a row in this stage averages {per_row_bytes / (1 << 20):.0f} MB. Byte-adaptive "
        f"batching has already shrunk the chunk to a row or two, and a row cannot be split, "
        f"so memory is bounded by the row itself from here. If this stage OOMs, the fix is "
        f"in the data: carry a path or handle instead of the blob, or decode it later.",
        PerformanceWarning,
        stacklevel=3,
    )
