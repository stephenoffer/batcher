"""What the conductor needs to know about a plan's size before it runs it.

Every function here answers from metadata alone — declared row counts, schema widths,
Kyber's per-breaker fan-out — so the answers are available *before* any I/O, early enough
to pick the streaming or out-of-core path instead of discovering the size by OOMing.

Unknowns are deliberately not optimistic: a source that cannot declare its size makes the
whole byte estimate unknown rather than partial, because understating is the direction
that runs a machine out of memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "DEFAULT_PARTITIONS",
    "declared_row_count",
    "distributed_hardware",
    "partitions_from_physical",
    "projected_input_bytes",
    "proven_empty_table",
]

# When the user leaves a knob unset, fill it from the same analyses Kyber and Carbonite
# already produce rather than a blind constant. This fallback is the historical default,
# used only when nothing about the data size is known.
DEFAULT_PARTITIONS = 16
_MIN_PARTITIONS = 4
_MAX_PARTITIONS = 4096


def _clamp_partitions(n: int) -> int:
    return max(_MIN_PARTITIONS, min(_MAX_PARTITIONS, n))


def distributed_hardware():
    """The cluster's `HardwareProfile` for planning, or `None` if the topology is unreadable.

    Isolated so the `dist` import stays lazy — a single-node run never touches Ray — and so
    a topology read that fails (Ray down, no worker nodes) degrades to `None`, leaving the
    optimizer to plan against the local machine rather than a fabricated cluster.

    Returns:
        The cluster profile, or `None`.
    """
    try:
        from batcher.dist.executors.ray_runtime.scaling import cluster_hardware_profile

        return cluster_hardware_profile()
    except Exception:  # pragma: no cover - Ray optional / topology unreadable
        return None


def partitions_from_physical(opt: PhysicalPlan) -> int | None:
    """Spill partition count implied by the optimized plan, or `None` if unsized.

    Reuses the per-breaker ``n_max_parallelism`` Kyber already computed (input rows /
    `target_rows_per_task`) — the same data-sized fan-out the distributed path uses — so
    out-of-core spilling shards by data volume instead of a blind constant.

    Floored at the machine's usable core count, because data volume alone answers only half
    the question. Kyber sizes this purely from rows, which is right for the distributed path
    where `clamp_workers` refits it to the cluster afterwards, but nothing refits it here. A
    40M-row aggregate at the default 4M rows/task is 10 partitions whether the box has 4
    cores or 128, and a spilled merge cannot use more cores than it has partitions.

    Args:
        opt: The optimized physical plan.

    Returns:
        A clamped partition count, or `None` when no breaker carries a fan-out.
    """
    widths = [op.bounds.n_max_parallelism for op in opt.ops if op.bounds.n_max_parallelism > 0]
    if not widths:
        return None
    return _clamp_partitions(max(max(widths), available_cpu_count()))


def projected_input_bytes(sources: list[Source], projections: dict[int, list[str]]) -> int:
    """Bytes the sources would occupy if resolved whole, from metadata alone.

    The in-memory path materializes every source before the engine starts, so this is the
    resident cost of *reading*, independent of what the query then computes. It is a row
    count times the projected schema's per-row width: no I/O, no scan.

    **An estimated row count counts here, and an exact one is not required.** That is the
    difference between this and `declared_row_count`, whose caller asks "did this read see
    the source whole?" — a question an estimate genuinely cannot answer, so it insists on
    exactness. This caller asks "will reading this fit in memory?", where the two failure
    modes are not symmetric: over-estimating routes a query out of core and costs latency,
    while having no estimate at all leaves it on the in-memory path and costs the process.

    Demanding exactness here made the guard depend on file format rather than on data size.
    Parquet carries a row count in its footer and was protected; CSV and JSON do not, so the
    identical query over the identical rows read `0` and ran unbounded — measured on a 64 MiB
    envelope, the Parquet copy routed to the out-of-core executor and the CSV copy did not.
    A source's own estimate (`statistics().row_count`, from a sampled scan) is the number it
    already computed for the optimizer; refusing it bought nothing.

    Args:
        sources: The plan's bound sources.
        projections: Pushed column projections, keyed by source index.

    Returns:
        The total byte estimate, or `0` when any source can offer no row count at all —
        which is not evidence of fitting, so the caller must fall back to its other signals.
    """
    from batcher.plan.types import schema_row_bytes

    total = 0.0
    for i, src in enumerate(sources):
        rows = declared_row_count(src)
        if rows is None:
            rows = _estimated_row_count(src)
        if rows is None or rows < 0:
            return 0
        try:
            schema = src.schema()
            projection = projections.get(i)
            if projection:
                schema = pa.schema([schema.field(schema.get_field_index(c)) for c in projection])
        except Exception:  # pragma: no cover - a source that cannot describe itself
            return 0
        total += rows * schema_row_bytes(schema)
    return int(total)


def _estimated_row_count(src: Source) -> int | None:
    """A source's *estimated* row count from its own statistics, or `None`.

    The number a format without a footer still knows — a CSV reader samples to estimate it
    for the optimizer, so it costs nothing extra here. Best-effort by construction: a source
    that cannot describe itself returns `None` and the caller falls back to its other
    signals, exactly as before.

    Args:
        src: The source to ask.

    Returns:
        The estimated rows, or `None`.
    """
    try:
        stats = src.statistics()
    except Exception:  # pragma: no cover - a source with no statistics at all
        return None
    rows = getattr(stats, "row_count", None) if stats is not None else None
    if rows is None or rows < 0:
        return None
    return int(rows)


def declared_row_count(src: Source) -> int | None:
    """The exact row count a source declares without a scan, or `None` if it cannot.

    Used to decide whether a read saw the source *whole*, so its distinct count may be
    learned. A source with no `row_count`, or one that raises, is unknown — the safe side,
    since an unverifiable "did I see everything?" must answer no.

    Args:
        src: The source to ask.

    Returns:
        The declared row count, or `None`.
    """
    fn = getattr(src, "row_count", None)
    if not callable(fn):
        return None
    try:
        n = fn()
    except Exception:  # pragma: no cover - a source that cannot count itself
        return None
    return int(n) if n is not None else None


def proven_empty_table(logical_opt: LogicalPlan, plan: LogicalPlan) -> pa.Table | None:
    """A typed, zero-row result when the optimizer proved the plan yields no rows.

    Kyber signals that proof by rewriting the root to a `Limit(input, 0)` — the only way the
    plan algebra can say "provably empty". The result is then fully determined by the output
    schema, so no source is read and the engine never runs.

    Args:
        logical_opt: The optimized logical plan.
        plan: The pre-optimization plan, whose schema types the empty result.

    Returns:
        A zero-row table, or `None` to execute normally.
    """
    from batcher.plan.logical import Limit

    if not (isinstance(logical_opt, Limit) and logical_opt.n == 0):
        return None
    inferred = plan.available_schema()
    return None if inferred is None else inferred.arrow.empty_table()
