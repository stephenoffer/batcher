"""What to do with a shard the device could not hold: make it smaller, not somebody else's.

Device memory is the constraint a GPU query actually runs into, and it is the one a *fixed*
shard count cannot answer. The shard count is chosen before the query runs, from an estimate,
and estimates are wrong in the direction that matters: a skewed key, a wider row than the
footer suggested, or a neighbouring tenant on the same device, and one shard does not fit while
every other one did.

Falling back to the host for that shard is correct and expensive — it hands the largest piece
of the work to the slowest executor. Subdividing it and running the pieces on the device is the
better answer, and it is *exact* here for the same reason the fan-out is exact at all: the
stage is mergeable, so a shard's partial and the concatenation of its halves' partials are the
same value. Splitting is a scheduling decision, not a semantic one.

The ladder is: subdivide (bounded, halving each round) → then the CPU engine. Only a failure
that reads as *memory* takes the first rung, because a deterministic error — an untranslatable
expression, a missing column — fails identically on a smaller shard, and retrying it in pieces
would pay N times to reach the same conclusion.
"""

from __future__ import annotations

import re

from batcher._internal.logging import note_suppressed

__all__ = [
    "device_peak_marker",
    "is_memory_failure",
    "measured_parts",
    "narrowed_schema",
    "peak_from_error",
    "plan_shard_count",
    "source_bytes",
    "split_descriptor",
]

#: Bytes per row assumed for a relation whose schema cannot be measured. Deliberately generous:
#: an under-estimate makes shards too large, and the recovery for that is a failed device.
_FALLBACK_ROW_BYTES = 128


def narrowed_schema(schema, projection: list[str] | None):
    """`schema` restricted to `projection` — the width a shard read through it actually holds.

    Packing prices a shard from its schema, so a projected read priced at the relation's full
    width looks several times larger than it is and the fan-out packs fewer tasks onto each
    board than fit. Both the aggregate and the join fan-outs narrow the same way, from here,
    rather than each keeping its own copy of a two-line rule.

    Args:
        schema: The source's Arrow schema, or `None` when it could not be read.
        projection: The columns the read returns, or `None` for all of them.

    Returns:
        The narrowed schema, or `schema` unchanged when there is nothing to narrow or a name
        is absent. A name the schema does not carry leaves it alone rather than raising:
        mispricing costs a slower fan-out, and this is not where a bad projection should
        surface.
    """
    if schema is None or projection is None:
        return schema
    try:
        import pyarrow as pa

        return pa.schema([schema.field(name) for name in projection])
    except Exception as exc:
        note_suppressed("dist", "narrow the shard packing schema to the projection", exc)
        return schema


def source_bytes(source, projection: list[str] | None = None) -> float:
    """Roughly how many bytes of device memory this source's rows would occupy.

    Estimated from the source's own row count and its *projected* schema, which is the width the
    read will actually produce — sizing a pruned read at the relation's full width would make
    every fan-out think it has several times the data it does.

    Args:
        source: The relation to size.
        projection: The columns that will be read, or `None` for all of them.

    Returns:
        The estimated byte size, or `0.0` when the source cannot say how many rows it has.
        Callers read `0.0` as "unknown" and keep whatever sizing they would have used.
    """
    try:
        rows = source.row_count()
    except Exception as exc:
        note_suppressed("dist", "read a source's row count for shard sizing", exc)
        return 0.0
    if not rows:
        return 0.0
    return float(rows) * _row_bytes(source, projection)


def _row_bytes(source, projection: list[str] | None) -> float:
    """The decoded width of one row, narrowed to `projection` where the schema is readable."""
    try:
        schema = source.schema()
    except Exception as exc:
        note_suppressed("dist", "read a source's schema for shard sizing", exc)
        return _FALLBACK_ROW_BYTES
    names = projection if projection is not None else schema.names
    total = 0.0
    for name in names:
        try:
            total += _field_bytes(schema.field(name).type)
        except KeyError:
            total += _FALLBACK_ROW_BYTES
    return max(total, 1.0)


def _field_bytes(dtype) -> float:
    """One column's bytes per row: its fixed width, or the optimizer's figure for a string."""
    import pyarrow as pa

    from batcher.config import active_config

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_binary(dtype):
        # A variable-width column has no per-row width until it is read. Using the optimizer's
        # own `row_bytes` keeps the routing decision and the shard sizing from disagreeing about
        # how big the same relation is.
        return float(active_config().optimizer.row_bytes)
    try:
        return max(dtype.bit_width / 8.0, 1.0)
    except (AttributeError, ValueError):
        return float(_FALLBACK_ROW_BYTES)


def plan_shard_count(total_bytes: float, gpu_count: int, device_bytes: float) -> int:
    """How many shards a fan-out over `total_bytes` should cut.

    Three bounds, and the interesting one is the third:

    * **memory** — every shard must fit a device with room for what the operators build on top
      of it, so a large relation needs at least `total / (device / expansion)` pieces;
    * **parallelism** — a fan-out wants at least one shard per device, or the fleet is idle;
    * **granularity** — and no shard smaller than `gpu_min_shard_bytes`, because below that the
      Ray dispatch that delivers a shard costs more than the shard's own compute.

    The third bound is the one that was missing. Shard count was `gpu_count x oversubscribe`
    whatever the input, so a 6M-row scan on sixteen devices was cut into sixty-four tasks of a
    hundred thousand rows each and spent 196 seconds doing 0.12 seconds of work. It also
    overrides the second: a relation too small to fill every device should run on one.

    Args:
        total_bytes: Estimated decoded size of what the fan-out will read, `0` when unknown.
        gpu_count: The cluster's live device count.
        device_bytes: One device's usable memory, `0` when it cannot be read.

    Returns:
        The shard count, at least 1. An unknown size keeps the device-count-driven sizing,
        which is what this path did before measuring anything.

    Examples:
        .. doctest::

            >>> from batcher.dist.gpu.shards import plan_shard_count
            >>> plan_shard_count(6 << 20, 16, 15e9)  # smaller than one shard's floor
            1
            >>> plan_shard_count(64 << 30, 16, 15e9)  # far larger than the fleet's memory
            64
    """
    from batcher.config import active_config

    dc = active_config().distributed
    devices = max(1, int(gpu_count))
    ceiling = devices * max(1, int(dc.gpu_shard_oversubscribe))
    if total_bytes <= 0:
        return ceiling
    floor_bytes = max(1, int(dc.gpu_min_shard_bytes))
    # Never cut a shard below the granularity floor. `ceil` rather than `floor` so a relation
    # smaller than one floor still gets exactly one shard rather than none.
    by_granularity = max(1, int(-(-total_bytes // floor_bytes)))
    wanted = min(devices, by_granularity)
    if device_bytes > 0:
        per_shard = device_bytes / max(1.0, float(dc.gpu_shard_expansion))
        wanted = max(wanted, int(-(-total_bytes // per_shard)))
    return max(1, min(ceiling, wanted, by_granularity))


#: Most pieces one round may divide a shard into. A shard that appears to want more than this
#: is one whose measurement is not to be trusted (a runaway cross product, a corrupt footer),
#: and dividing by a thousand would cost a thousand reads to discover that.
MAX_MEASURED_PARTS = 16

#: How a worker reports what its device had actually drawn, inside the error that reports the
#: overflow. Carried in the message *text* for the same reason `_MEMORY_MARKERS` are matched
#: that way: the exception reaches the driver through Ray, which re-raises it as its own
#: wrapper type and leaves only the message intact.
_PEAK_PATTERN = re.compile(r"\[bt-device-peak (\d+)/(\d+)\]")

#: Substrings that identify an allocation failure across the layers a GPU task crosses — RMM
#: and the CUDA driver underneath cuDF, the C++ allocator, and Python's own `MemoryError`.
#: Matched on text because the exception arrives through Ray, which re-raises a task's error
#: as its own wrapper type and leaves only the message intact.
_MEMORY_MARKERS = (
    "out of memory",
    "out_of_memory",
    "bad_alloc",
    "cudaerrormemoryallocation",
    "rmm",
    "memoryerror",
    "insufficient memory",
    "resource_exhausted",
    # RMM's `PoolMemoryResource` refuses past its ceiling with this and no other wording, and
    # none of the markers above appear in it. It arrives at the driver wrapped by Ray as a
    # `RayTaskError`, so the exception *type* that would otherwise have carried the word "rmm"
    # is gone by the time it is classified — leaving the one overflow that a bounded pool
    # produces by design as the one overflow the subdivision ladder did not recognize. It read
    # as a deterministic error and the shard went straight to the CPU engine.
    "maximum pool size exceeded",
    # cuDF's spill manager, when `spill_to_host` is on and the host has nothing left either.
    "failed to allocate",
)


def is_memory_failure(exc: BaseException) -> bool:
    """Whether `exc` reads as the device running out of memory.

    A memory failure is worth retrying on smaller pieces; a deterministic one is not, and
    treating the two alike would multiply the cost of every real error by the split factor
    while changing none of their outcomes.

    Args:
        exc: The error a shard's task raised.

    Returns:
        True when the error names an allocation failure anywhere in its message chain.
    """
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        text += f" {type(cause).__name__} {cause}".lower()
    return any(marker in text for marker in _MEMORY_MARKERS)


def device_peak_marker() -> str:
    """This process's device high-water mark, formatted to travel inside an error message.

    Called on the **worker**, at the moment its shard overflowed. That is the only place the
    figure exists: the driver deciding how far to subdivide has no device, so asking its own
    allocator — which is what happened — returns nothing on every distributed run and the
    subdivision silently falls back to blind halving. Reading it here and appending it to the
    error is what carries the measurement to the one process that needs it.

    Returns:
        `"[bt-device-peak <peak>/<pool>]"`, or `""` when the allocator was not asked to keep
        statistics or there is no device — where the caller appends nothing and the driver
        halves blindly, exactly as before.
    """
    from batcher.carbonite.accel import device_allocator_state

    try:
        state = device_allocator_state()
    except Exception as exc:  # pragma: no cover - a diagnostic must not mask the real error
        note_suppressed("dist", "read the device high-water mark for a failed shard", exc)
        return ""
    peak, pool = int(state.get("peak_bytes", 0)), int(state.get("pool_bytes", 0))
    return f" [bt-device-peak {peak}/{pool}]" if peak > 0 and pool > 0 else ""


def peak_from_error(exc: BaseException | None) -> tuple[int, int]:
    """The `(peak, pool)` bytes a worker reported inside its overflow, or `(0, 0)`.

    Args:
        exc: The error a shard's task raised, as the driver received it.

    Returns:
        The measured high-water mark and the pool it exceeded. `(0, 0)` when the error carries
        no marker — an older worker, an allocator without statistics, or a failure that was
        never about memory.
    """
    if exc is None:
        return (0, 0)
    for text in (str(exc), str(exc.__cause__ or ""), str(exc.__context__ or "")):
        found = _PEAK_PATTERN.search(text)
        if found:
            return (int(found.group(1)), int(found.group(2)))
    return (0, 0)


def measured_parts(default: int = 2, exc: BaseException | None = None) -> int:
    """How many pieces a shard that just overflowed should be divided into, from what it drew.

    Halving is the right *blind* answer, but it is only ever right by accident: a shard that
    peaked at eight times the pool takes three failed rounds to reach a size that fits, and
    each of those rounds re-reads the whole shard from storage. When the allocator was asked
    to keep statistics, the high-water mark says how far over the shard actually went, and the
    factor that clears it in one round is arithmetic rather than a search.

    The measurement is read **from the error** first, and only then from this process. That
    ordering is the whole point: the subdivision is decided on the driver, the overflow happened
    on a worker, and a driver asking its own allocator gets nothing on every distributed run —
    so the "measured" factor was measured only on the single-node dispatch path, where the
    decider and the device are the same process. Both callers now get the same answer.

    Args:
        default: The factor to use when nothing was measured — the blind halving.
        exc: The error that reported the overflow, whose message may carry the worker's own
            high-water mark (see `device_peak_marker`). `None` falls back to this process.

    Returns:
        The number of pieces to divide into, `default` whenever no peak and ceiling could be
        compared. An unmeasured overflow keeps the old behavior rather than being divided
        against a fabricated figure.

    Examples:
        .. doctest::

            >>> from batcher.dist.gpu import measured_parts
            >>> measured_parts()  # nothing measured on a host with no device
            2
    """
    peak, ceiling = peak_from_error(exc)
    if peak <= 0 or ceiling <= 0:
        from batcher.carbonite.accel import device_allocator_state

        state = device_allocator_state()
        peak, ceiling = int(state.get("peak_bytes", 0)), int(state.get("pool_bytes", 0))
    if peak <= 0 or ceiling <= 0 or peak <= ceiling:
        return default
    return max(default, min(MAX_MEASURED_PARTS, -(-peak // ceiling)))


def split_descriptor(descriptor: dict, parts: int = 2) -> list[dict]:
    """Divide one partition descriptor into up to `parts` smaller ones.

    Splitting the *descriptor* rather than the data is what keeps this cheap: a split-manifest
    names the files a worker should read, so dividing the manifest divides the read without the
    driver touching a row. A batch-list descriptor (an in-memory source) divides its batches,
    which are already on the driver by construction.

    Args:
        descriptor: A descriptor from `partition_descriptors`.
        parts: How many pieces to divide it into.

    Returns:
        The pieces, or `[descriptor]` when it cannot be divided further — a single split or a
        single batch is already the smallest thing a worker can be asked to read.
    """
    for key in ("splits", "batches"):
        items = descriptor.get(key)
        if items is None:
            continue
        if len(items) < 2:
            return [descriptor]
        chunks = _chunk(items, min(parts, len(items)))
        return [{**descriptor, key: chunk} for chunk in chunks]
    return [descriptor]


def _chunk(items: list, parts: int) -> list[list]:
    """`items` divided into `parts` contiguous, near-equal pieces."""
    step = -(-len(items) // parts)
    return [items[i : i + step] for i in range(0, len(items), step)]


def run_subdivided(descriptor, run, *, parts: int, rounds: int, split=None, cause=None):
    """Run `descriptor` in pieces, halving further while the device still cannot hold one.

    `run(descriptor)` executes one piece and returns its Arrow partial (or `None` when empty).
    The pieces' partials are concatenated, which is exactly the shard's own partial because the
    stage is mergeable — subdividing changes how the work is scheduled and not what it computes.

    Args:
        descriptor: The shard that did not fit. A single descriptor for a chain; whatever unit
            `split` divides for a caller whose shard is more than one relation.
        run: Executes one descriptor, returning its partial.
        parts: How many pieces to divide into each round.
        rounds: How many times to subdivide before giving up.
        split: `split(item, parts) -> [item, ...]`, defaulting to `split_descriptor`. A plan
            tree's shard is a *list* of descriptors, only one of which may be divided — the
            others are the relations every worker must see whole — so what "smaller" means is
            the caller's to say rather than this function's to assume.
        cause: The error that reported the overflow. Its message carries the worker's own
            device high-water mark, which is what lets the first division be sized rather than
            guessed — this function runs on the driver, which has no device to measure.

    Returns:
        The concatenation of the pieces' partials, or `None` when every piece was empty.

    Raises:
        Exception: The last error, once the rounds are spent or a piece fails for a reason that
            is not memory — so the caller can fall back to the host with an accurate cause.
    """
    import pyarrow as pa

    divide = split if split is not None else split_descriptor
    parts = measured_parts(parts, cause)
    pending = [((i,), d) for i, d in enumerate(divide(descriptor, parts))]
    if len(pending) == 1:
        # Chained to the overflow that got here. The caller falls back to the host on this
        # error and reports it as the reason, and a bare `MemoryError` erased the device-side
        # detail — the driver, the allocator, the byte figure — leaving "cannot be divided
        # further" as the whole account of why a GPU query ran on the CPU.
        raise MemoryError("a shard of one split cannot be divided further") from cause
    # Keyed by position, and sorted before concatenating. A row-local chain's shard is a
    # contiguous slice of the source and its pieces are contiguous slices of that, so their
    # order IS part of the answer; appending them as they happen to finish would reorder the
    # result of every filter that ever met a device too small for it.
    done: dict[tuple, object] = {}
    for _ in range(max(1, rounds)):
        failed: list[tuple] = []
        last: BaseException | None = None
        for key, piece in pending:
            try:
                out = run(piece)
            except Exception as exc:
                if not is_memory_failure(exc):
                    raise
                last = exc
                # Re-measured per piece, from *this* failure. A piece that overflowed by eight
                # times is divided by eight rather than halved four rounds in a row, and each
                # of those rounds would have re-read the piece from storage to learn nothing.
                smaller = divide(piece, measured_parts(parts, exc))
                if len(smaller) == 1:
                    raise
                note_suppressed("dist", "gpu shard piece; subdividing further", exc)
                failed.extend(((*key, j), d) for j, d in enumerate(smaller))
                continue
            if out is not None and out.num_rows:
                done[key] = out
        if not failed:
            # Lexicographic on the position path, so a piece that was subdivided again sorts
            # inside the piece it came from: (1,) < (1, 0) < (1, 1) < (2,).
            ordered = [done[k] for k in sorted(done)]
            return pa.concat_tables(ordered) if ordered else None
        pending = failed
    raise last or MemoryError("gpu shard did not fit after subdividing")


class ShardReport:
    """Counts how a fan-out's shards actually ran, and says so when it was not the plain way.

    A fan-out that recovers is doing its job, and a fan-out that recovers *constantly* is a
    cluster telling you something — devices too small for the shard size, or a spot pool being
    reclaimed faster than the work finishes. Both produce the same answer as a clean run, so
    without an event the difference is invisible: the query is simply slower, for no reason the
    operator can see.

    Silent when nothing degraded, so the ordinary case adds no noise.

    The one exception is the *packing*: a fan-out that granted each shard a fraction of a device
    reports that even on a clean run, because it is the difference between the shards running
    together and running one at a time, and no other signal distinguishes them. A run that
    subdivides constantly *and* packed aggressively is one specific, fixable misconfiguration —
    `gpu_shard_expansion` set below what the chain actually materializes — and seeing the two
    counts side by side is what makes it diagnosable rather than merely slow.
    """

    __slots__ = ("packing", "recovered", "shards", "stage", "subdivided")

    def __init__(self, stage: str, shards: int, packing: object = None) -> None:
        """Start a report for `stage` over `shards` shards.

        Args:
            stage: The fan-out's name, as it appears in the event stream.
            shards: How many shards were submitted.
            packing: The `TaskPacking` the shards were scheduled with, or `None` when the caller
                did not pack (a fan-out on the old whole-device path reports exactly as before).
        """
        self.stage = stage
        self.shards = shards
        self.subdivided = 0
        self.recovered = 0
        self.packing = packing

    def note_subdivided(self) -> None:
        """One shard did not fit its device and was run in pieces."""
        self.subdivided += 1

    def note_recovered(self) -> None:
        """One shard's device was lost and the CPU engine produced its partial instead."""
        self.recovered += 1

    def publish(self) -> None:
        """Emit the summary if anything is worth saying; do nothing if the run was plain."""
        packed = self.packing is not None and getattr(self.packing, "packed", False)
        if not (self.subdivided or self.recovered or packed):
            return
        from batcher._internal import events

        payload: dict = {
            "name": self.stage,
            "event": "shard_degraded" if (self.subdivided or self.recovered) else "shard_packed",
            "shards": self.shards,
            "subdivided": self.subdivided,
            "recovered_on_cpu": self.recovered,
        }
        if self.packing is not None:
            payload["packing"] = self.packing.as_dict()
        events.publish(events.RECOVERY, **payload)
