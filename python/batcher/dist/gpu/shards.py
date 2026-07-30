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

from batcher._internal.logging import note_suppressed

__all__ = ["is_memory_failure", "measured_parts", "split_descriptor"]

#: Most pieces one round may divide a shard into. A shard that appears to want more than this
#: is one whose measurement is not to be trusted (a runaway cross product, a corrupt footer),
#: and dividing by a thousand would cost a thousand reads to discover that.
MAX_MEASURED_PARTS = 16

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


def measured_parts(default: int = 2) -> int:
    """How many pieces a shard that just overflowed should be divided into, from what it drew.

    Halving is the right *blind* answer, but it is only ever right by accident: a shard that
    peaked at eight times the pool takes three failed rounds to reach a size that fits, and
    each of those rounds re-reads the whole shard from storage. When the allocator was asked
    to keep statistics, the high-water mark says how far over the shard actually went, and the
    factor that clears it in one round is arithmetic rather than a search.

    Args:
        default: The factor to use when nothing was measured — the blind halving.

    Returns:
        The number of pieces to divide into, `default` whenever the device did not report a
        peak or a ceiling to compare it against. An unmeasured device keeps the old behavior
        rather than being divided against a fabricated figure.

    Examples:
        .. doctest::

            >>> from batcher.dist.gpu import measured_parts
            >>> measured_parts()  # nothing measured on a host with no device
            2
    """
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


def run_subdivided(descriptor: dict, run, *, parts: int, rounds: int):
    """Run `descriptor` in pieces, halving further while the device still cannot hold one.

    `run(descriptor)` executes one piece and returns its Arrow partial (or `None` when empty).
    The pieces' partials are concatenated, which is exactly the shard's own partial because the
    stage is mergeable — subdividing changes how the work is scheduled and not what it computes.

    Args:
        descriptor: The shard that did not fit.
        run: Executes one descriptor, returning its partial.
        parts: How many pieces to divide into each round.
        rounds: How many times to subdivide before giving up.

    Returns:
        The concatenation of the pieces' partials, or `None` when every piece was empty.

    Raises:
        Exception: The last error, once the rounds are spent or a piece fails for a reason that
            is not memory — so the caller can fall back to the host with an accurate cause.
    """
    import pyarrow as pa

    parts = measured_parts(parts)
    pending = [((i,), d) for i, d in enumerate(split_descriptor(descriptor, parts))]
    if len(pending) == 1:
        raise MemoryError("a shard of one split cannot be divided further")
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
                smaller = split_descriptor(piece, parts)
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
