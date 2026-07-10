"""Post-execution column-statistics learning (Core measures, Kyber persists).

After a query runs, the engine has the base sources' batches in hand; measuring each
column's ndv / quantiles / average byte width / most-common-values once and recording
them into the `MetadataHub` is what lets the *next* run's optimizer size joins,
aggregations, and broadcasts from learned numbers instead of Selinger guesses. The
work is gated to columns not already known, so a column's O(rows) sketch build happens
at most once. Best-effort throughout: a measurement failure never affects a result.

Extracted from `orchestration` along the measurement seam to keep the conductor module
within the size budget; the conductor calls these on its single-node and UDF paths.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.config import active_config
from batcher.io.source import Source, iter_source
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Not
from batcher.plan.logical import Aggregate, Filter, Join, LogicalPlan
from batcher.plan.visitor import walk

__all__ = ["collect_source_metadata", "learn_column_stats", "ndv_columns", "seed_column_ndv"]


# Row cap for the driver-side column-stat sample (≈ a couple of Parquet row-groups).
# Enough for usable ndv/quantile sketches; small enough that learning never re-scans a
# large input on the driver.
_STATS_SAMPLE_ROWS = 1 << 18


def _stats_sample(src: Source) -> list[pa.RecordBatch]:
    """A bounded row sample of `src` for column-stat learning — NOT the whole source.

    `collect_source_metadata` runs on the driver after a query; on the distributed/UDF
    paths it has no scanned batches in hand and so samples the base source here. Reading
    the *whole* source would re-scan it single-node on the driver — which on a large
    distributed input dwarfs the query itself (sf100: a 140 GB driver re-read that hung
    for minutes *after* a ~60 s distributed agg). The actual cardinalities of that run are
    already learned from the worker metrics; these column sketches are an approximate
    prior the estimator refines across runs, so a bounded sample is the right trade.

    For a splittable Parquet source the sample is read through the coalesced, multi-thread
    dataset scanner (`_fast_sample`) — the naive per-file driver read of even the bounded
    262 k-row sample is ~0.8 MB/s on high-latency object storage (measured 45 s for a
    single sf10 sample, which then blocks the query *return* after the distributed work is
    already done). The fast reader cuts that to ~0.4 s. Any non-splittable source (or a
    read failure) falls back to the lazy `iter_source` read; `iter_source` stops after the
    first batches past the row cap (an in-memory source is already resident and small)."""
    fast = _fast_sample(src)
    if fast is not None:
        return fast
    out: list[pa.RecordBatch] = []
    n = 0
    for b in iter_source(src, None, None):
        out.append(b)
        n += b.num_rows
        if n >= _STATS_SAMPLE_ROWS:
            break
    return out


def _fast_sample(src: Source) -> list[pa.RecordBatch] | None:
    """The bounded sample read through the coalesced Parquet row-group scanner, or `None`.

    Reuses the distributed scan reader (pre-buffered, 32-IO-thread pyarrow dataset scan)
    over just the leading row-group splits needed to cover the row cap — the same fast
    path a worker uses — instead of the naive single-stream driver read. Returns `None`
    for a non-row-group source (in-memory, CSV, one whole-file split) or any failure, so
    the caller falls back to `iter_source`."""
    try:
        from batcher.io.splits import RowGroupSplit

        splits = src.splits()
        if not splits or not all(isinstance(s, RowGroupSplit) for s in splits):
            return None
        # Take leading row-groups until their combined rows cover the cap (at least one).
        chosen: list = []
        rows = 0
        for s in splits:
            chosen.append(s)
            rows += s.row_count() or 0
            if rows >= _STATS_SAMPLE_ROWS:
                break
        from batcher.dist.executors.scan_read import _read_split_batches

        out: list[pa.RecordBatch] = []
        n = 0
        for b in _read_split_batches(chosen, None, None):
            out.append(b)
            n += b.num_rows
            if n >= _STATS_SAMPLE_ROWS:
                break
        return out or None
    except Exception:  # any read/scan failure → caller falls back to iter_source
        return None


def collect_source_metadata(hub, sources: list[Source]) -> None:
    """Record per-column ndv/quantiles from the base sources (Core collects).

    The UDF and distributed paths don't surface their scanned batches the way the
    native path hands `resolved` to `learn_column_stats`, so this samples the base
    sources directly (see `_stats_sample` — bounded, never a whole-source driver scan).
    It is gated on the cheap `Source.schema` — a source is only read when it has a
    not-yet-measured column — so a file is never re-scanned once its columns are
    learned. Best-effort: learning never breaks a query.
    """
    if hub is None:
        return
    from batcher import kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(kyber.NDV_KEY, {}))
        resolved = [
            _stats_sample(src) for src in sources if any(c not in known for c in src.schema().names)
        ]
        if resolved:
            learn_column_stats(hub, resolved)
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def ndv_columns(plan: LogicalPlan) -> set[str]:
    """The columns whose distinct count actually steers a plan decision.

    The estimator reads `ndv` in exactly three places: a join's key cardinality
    (`|L||R| / max(ndv_L, ndv_R)`), a `GROUP BY`'s output size (the product of its key
    `ndv`s), and an equality/`IN` predicate's `1/ndv` selectivity. Sketching anything else
    is wasted work — and at scale it is the difference between seeding and skipping: a
    60M-row `lineitem`'s sixteen columns blow any sane cell budget, while its three join
    keys fit comfortably.

    Names are taken from the plan as written, so a join key renamed by an intervening
    projection is simply not matched against the source schema and falls back to the
    post-run learner. Conservative in the right direction: a missed column costs a worse
    first plan, never a wrong answer.
    """
    wanted: set[str] = set()
    for node in walk(plan):
        if isinstance(node, Join):
            wanted.update(node.left_keys)
            wanted.update(node.right_keys)
        elif isinstance(node, Aggregate):
            wanted.update(k.expr.name for k in node.group_keys if isinstance(k.expr, Col))
        elif isinstance(node, Filter):
            wanted.update(_equality_columns(node.predicate))
    return wanted


def _equality_columns(expr: Expr) -> set[str]:
    """Columns compared by equality or `IN` anywhere in a predicate (they use `1/ndv`)."""
    out: set[str] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Binary):
            if node.op in ("eq", "ne"):
                out.update(e.name for e in (node.left, node.right) if isinstance(e, Col))
            stack.extend((node.left, node.right))
        elif isinstance(node, InList):
            if isinstance(node.input, Col):
                out.add(node.input.name)
        elif isinstance(node, Not):
            stack.append(node.input)
    return out


def seed_column_ndv(hub, sources: list[Source], plan: LogicalPlan | None = None) -> None:
    """Sketch base-source distinct counts *before* the optimizer runs (Core measures).

    `ndv` is the one statistic no file footer carries, and the two places Kyber needs it
    most — join cardinality (`|L||R| / max(ndv_L, ndv_R)`) and `GROUP BY` sizing (the
    product of the key `ndv`s) — degrade to `max(|L|, |R|)` and `0.1 · rows` without it.
    Learned *after* a run (`learn_column_stats`) it sharpens every later query, but the
    first execution plans blind: TPC-H Q9 cold joins `lineitem ⋈ partsupp ⋈ orders` into
    2.0 GB and 2.6 GB intermediates before reaching the 5%-selective `part` filter, while
    the same query re-run — `ndv` now known — takes the selective join first and
    materializes 32 MB.

    Seeding closes that gap for **resident** sources (already in memory, so sketching
    costs no I/O): an HLL-only pass over the not-yet-measured columns, ~12 ms for five
    columns of a 6M-row `lineitem`. A non-resident source is skipped — re-reading it here
    only to sketch would double the query's I/O — and keeps learning from the post-run
    pass, which sees its scanned batches for free.

    Only the columns `ndv_columns(plan)` names are sketched — the join keys, group keys
    and equality-predicate columns the estimator actually reads. Without that restriction
    a 60M-row `lineitem`'s sixteen columns exceed `optimizer.ndv_sketch_max_cells` and the
    source is skipped entirely, which is precisely the scale where a blind plan hurts most
    (TPC-H Q8 at sf10 peaks at 23 GB without it).

    Idempotent and self-limiting: a measured column lands in `known` and is never
    re-sketched, so this costs one hub read per query thereafter. The estimates are
    HyperLogLog (~1% error) recorded through the same `SKETCH`-provenance channel as the
    post-run learner, so they can never answer an exact `count_distinct`. Best-effort: a
    failure here never affects the query result.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        wanted = ndv_columns(plan) if plan is not None else None
        known = set(kyber.load_learned_stats(hub).get(kyber.NDV_KEY, {}))
        max_cells = active_config().optimizer.ndv_sketch_max_cells
        ndv_all: dict[str, float] = {}
        for src in sources:
            if not getattr(src, "resident", False):
                continue
            cols = [
                c
                for c in src.schema().names
                if c not in known and c not in ndv_all and (wanted is None or c in wanted)
            ]
            rows = src.row_count() or 0
            if not cols or rows * len(cols) > max_cells:
                continue
            ndv_all.update(core.column_ndv(src.read(projection=cols), cols))
        if ndv_all:
            kyber.record_column_stats(hub, ndv_all, {})
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def learn_column_stats(hub, resolved: list[list[pa.RecordBatch]]) -> None:
    """Measure per-column ndv/quantiles from the just-scanned input and record them.

    Gated to columns not already known, so the O(rows) sketch build happens at most
    once per column — a bounded, one-time cost that sharpens every later plan. Core
    measures (`core.column_statistics`); Kyber persists/consumes. Best-effort: a
    failure here never affects the query result.

    The "already measured" marker is the *average byte width*, not the distinct count:
    `column_statistics` records one for every column it touches (numeric or not), whereas
    `ndv` alone is also written by the cheaper pre-optimize `seed_column_ndv`. Gating on
    `ndv` would let that seeding suppress this pass, losing the quantile grids and
    most-common-values it is the only source of.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(kyber.AVG_BYTES_KEY, {}))
        min_frac = active_config().optimizer.cardinality.mcv_min_fraction
        ndv_all: dict[str, float] = {}
        quant_all: dict[str, dict[str, list[float]]] = {}
        bytes_all: dict[str, float] = {}
        mcv_all: dict[str, dict[str, float]] = {}
        for batches in resolved:
            if not batches:
                continue
            cols = [c for c in batches[0].schema.names if c not in known]
            if not cols:
                continue
            ndv, quants, avg_bytes = core.column_statistics(batches, cols)
            ndv_all.update(ndv)
            quant_all.update(quants)
            bytes_all.update(avg_bytes)
            total = sum(b.num_rows for b in batches)
            # MCV clears `min_frac` only on low-cardinality columns (ndv ≲ 1/min_frac);
            # skip the per-row Misra-Gries scan on keys/high-ndv columns (always empty).
            mcv_cols = [c for c in cols if ndv.get(c, 1e18) <= 1.0 / min_frac]
            for col_name, hits in core.heavy_hitters(batches, mcv_cols, min_frac).items():
                if total > 0 and hits:
                    mcv_all[col_name] = {str(v): n / total for v, n in hits}
        if ndv_all or quant_all or bytes_all or mcv_all:
            kyber.record_column_stats(hub, ndv_all, quant_all, bytes_all, mcv_all)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
