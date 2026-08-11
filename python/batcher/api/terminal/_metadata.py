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

import random

import pyarrow as pa

from batcher._internal.logging import get_logger, note_suppressed
from batcher.config import active_config
from batcher.io.base._bad_rows import measuring
from batcher.io.source import Source, iter_source
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Not
from batcher.plan.logical import Aggregate, Filter, Join, LogicalPlan
from batcher.plan.source_stats import source_stats_key
from batcher.plan.visitor import walk

__all__ = [
    "collect_source_metadata",
    "learn_column_stats",
    "learnable_columns",
    "ndv_columns",
    "seed_column_ndv",
]

_log = get_logger("metadata")

# Picks each sampled batch's window offset (`_sketch_sample`). Seeded, so a run's learned
# statistics — and therefore the plans they steer — are reproducible.
_rng = random.Random(0x5EED)


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
    # Under `measuring()`: this read meets the same malformed records the data read did,
    # and counting them again would inflate the very metric that says how much data the job
    # quietly dropped. Tolerance still applies — a bad record must not fail the sample —
    # only the tally is suppressed.
    with measuring():
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
        learned = kyber.load_learned_stats(hub)
        sampled: list[list[pa.RecordBatch]] = []
        keep: list[Source] = []
        for src in sources:
            source_key = source_stats_key(src)
            if source_key is None:
                continue
            known = set(kyber.columns_for(learned, kyber.NDV_KEY, source_key))
            if any(c not in known for c in src.schema().names):
                sampled.append(_stats_sample(src))
                keep.append(src)
        if sampled:
            learn_column_stats(hub, sampled, keep)
    except Exception as exc:  # pragma: no cover - learning must never break execution
        note_suppressed("api", "learn column statistics", exc)


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

    **`ephemeral` sources are skipped, and that is what makes the idempotence above true.**
    An adaptive stage boundary hands the next stage its intermediate wrapped as an
    in-memory source, and an in-memory source is keyed by object identity — so a stage
    source's key is *new on every execution*. Nothing it recorded could ever be read again,
    and three things followed from recording it anyway, each worse than the last: the
    sketch was recomputed every run (TPC-H Q8 at sf10 re-sketched 807k rows per collect,
    and 280M on the first); the learned store grew by one dead `obj:<id>` entry per
    execution, without bound; and because a column absent from the store is by definition
    "measured for the first time", `record_column_stats` bumped the learned **generation**
    every single execution — which is the plan cache's key, so the cache never once hit and
    every run re-planned from scratch (Q8 130 ms, Q2 50 ms, against DuckDB's 84 ms and
    46 ms for the *whole query*). A statistic keyed by an identity that cannot recur is not
    a statistic; it is a leak that also reports itself as news.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        wanted = ndv_columns(plan) if plan is not None else None
        learned = kyber.load_learned_stats(hub)
        max_cells = active_config().optimizer.ndv_sketch_max_cells
        for src in sources:
            if not getattr(src, "resident", False) or getattr(src, "ephemeral", False):
                continue
            source_key = source_stats_key(src)
            if source_key is None:
                continue  # an unkeyable source: its stats cannot be told apart from another's
            # "Already measured" is a question about *this* source, not about any column
            # anywhere: a column named `id` measured on another table says nothing here.
            known = set(kyber.columns_for(learned, kyber.NDV_KEY, source_key))
            cols = [
                c for c in src.schema().names if c not in known and (wanted is None or c in wanted)
            ]
            rows = src.row_count() or 0
            if not cols or rows * len(cols) > max_cells:
                continue
            batches = src.read(projection=cols)
            ndv = core.column_ndv(batches, cols)
            # Heavy hitters, from the same batches, for the same reason the ndv is seeded:
            # the data is already resident, so measuring costs no I/O — and skew is the one
            # distribution the estimator cannot approximate at all without it.
            #
            # Under uniformity an equality is `1/ndv`, which is exactly wrong on a skewed
            # key. Measured on 100,000 rows with one value at 50%: `k = 7` estimated at 20
            # rows against a true 49,868, a **2,487x** under-estimate, and a Zipf column was
            # 139x under. That estimate sizes the filter, then the join built on its output,
            # then that join's memory envelope — so a cold query over a hot key mis-sizes
            # every stage below it. `learn_column_stats` measures this today, but only
            # *after* a query of the shape has run, so the first one plans blind.
            #
            # Over a **bounded sample**, exactly as the post-run pass does: Misra-Gries is
            # ~56 ns a cell against the ndv sketch's ~4, so sketching a whole large source
            # here would add to every cold query what `_STATS_SAMPLE_ROWS` exists to cap. A
            # frequency is a fraction of rows and a uniform sample preserves it, which is
            # what makes the bound sound rather than merely cheap.
            mcv: dict[str, dict[str, float]] = {}
            sample = _sketch_sample(batches)
            sampled_rows = sum(b.num_rows for b in sample)
            if sampled_rows > 0:
                min_frac = active_config().optimizer.cardinality.mcv_min_fraction
                for col_name, hits in core.heavy_hitters(sample, cols, min_frac).items():
                    if hits:
                        mcv[col_name] = {str(v): n / sampled_rows for v, n in hits}
            if ndv or mcv:
                # `avg_bytes` is deliberately left empty: `learn_column_stats` gates its own
                # "already known" check on that key precisely so seeding cannot suppress the
                # post-run pass, which is the only source of quantile grids. Writing one
                # here would silently switch that pass off for the column.
                kyber.record_column_stats(hub, ndv, {}, mcv=mcv, source_key=source_key)
    except Exception as exc:  # pragma: no cover - learning must never break execution
        note_suppressed("api", "learn column NDV", exc)


def _learn_row_bytes(hub, resolved, sources) -> None:
    """Record `nbytes / rows` for every column of every keyable source in `resolved`.

    The cheap half of column learning, and the half that was missing. `learn_column_stats`
    restricts its sketches to the columns a later plan could consult a *distribution* for, on
    the sound reasoning that a KLL grid for a column nothing filters on is pure loss. A byte
    width is the exception: `StatsEstimator.row_width` sums widths over every **output**
    column, so the payload columns no predicate mentions — the embedding, the document, the
    frame — are both the widest in the row and the ones never measured.

    An ephemeral source is skipped for the reason the sketch pass skips it: its key does not
    survive the execution, so anything filed under it is unreadable by a later query.

    Best-effort and silent: a source that cannot report a width simply contributes none.
    """
    from batcher import kyber

    if not resolved:
        return
    # What is already on file, so a steady-state query writes nothing. `merge_column_table` is
    # a whole-table read-modify-write, and this runs on *every* execution rather than being
    # gated by the sketch pass's "already measured" marker — so without this check a served
    # workload would pay that write per query forever to re-record the same numbers.
    known = kyber.load_learned_stats(hub)
    for i, batches in enumerate(resolved):
        if not batches:
            continue
        source = sources[i] if sources is not None and i < len(sources) else None
        if source is None or getattr(source, "ephemeral", False):
            continue
        try:
            source_key = source_stats_key(source)
            if source_key is None:
                continue
            rows = sum(b.num_rows for b in batches)
            if rows <= 0:
                continue
            on_file = kyber.columns_for(known, kyber.ROW_BYTES_KEY, source_key)
            names = [n for n in batches[0].schema.names if n not in on_file]
            if not names:
                # Every column of this source already has a width, so there is nothing to
                # learn and the measurement below is pure cost. It is not a small one:
                # `Array.nbytes` walks a column's buffers and is charged per batch *and* per
                # column, so `lineitem` at scale 1 (49 batches x 16 columns) is a 5.1 ms sweep
                # — 18% of TPC-H q6's entire wall time, paid on every execution to re-derive
                # numbers already on file. The write was guarded against that; the
                # measurement was not.
                #
                # Gating on the columns rather than on a marker keeps the loop live where it
                # can still learn: a projection that returns a column not measured before, a
                # new source object (an in-memory source is keyed per instance), or a first
                # run all fall through and measure. What it gives up is re-deriving a width
                # for a *file* source whose bytes changed under an unchanged path — and a
                # width is a cost input rather than an answer, so a stale one costs plan
                # quality, never correctness. This is the same trade the sketch pass makes
                # with its "already measured" marker.
                continue
            # Only the columns with nothing on file, so every measurement here is one the
            # store does not have. `is_material_change` guarded the write when this measured
            # every column on every run; with the gate above it would be checking a prior
            # that is `None` by construction, which is worse than not checking it.
            widths: dict[str, float] = {}
            for name in names:
                total = sum(b.column(name).nbytes for b in batches)
                if total > 0:
                    widths[name] = total / rows
            if widths:
                kyber.record_column_row_bytes(hub, widths, source_key=source_key)
        except Exception:
            # Per source, and swallowed here rather than by the caller's `try`. This runs
            # *before* the sketch pass, so letting one unreadable source's width escape would
            # cost every source its quantiles and most-common-values as well — a failure in
            # the cheap half taking the expensive half down with it.
            _log.debug("row-width learning failed for one source", exc_info=True)


def learnable_columns(plan: LogicalPlan) -> set[str]:
    """The columns whose measured statistics a later plan of this shape could actually read.

    The union of the two things the estimator consults: `ndv_columns` (join keys, group
    keys, equality predicates — the distinct counts) and the columns any `Filter` mentions
    (the quantile grids and most-common-values, which drive range and equality selectivity).
    A column outside that union has no consumer; sketching it is pure loss.
    """
    from batcher.api.source_stats import column_bounds_needed

    return ndv_columns(plan) | column_bounds_needed(plan)


# How many rows the *expensive* sketches (KLL quantiles, Misra-Gries most-common-values)
# read, however large the input is. See `_sketch_sample`.
#
# 262,144 rows keeps a quantile grid well inside the KLL's own ~1% rank error and resolves a
# most-common-value down to a fraction of a percent — far finer than the 5% floor that
# decides whether a key is hot. Doubling it would buy accuracy neither consumer can use.
_SKETCH_SAMPLE_ROWS = 262_144


def _sketch_sample(batches: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
    """A bounded, order-stratified sample of `batches` — for the sketches too costly to run
    over everything.

    The three learned statistics do not cost the same. HyperLogLog reads a cell in ~4 ns, so
    distinct counts are measured over every row. The KLL quantile sketch and the Misra-Gries
    most-common-value summary cost ~56 ns a cell between them, and running *those* over the
    whole input is what made the first run of a 3 ms query take 140 ms — the engine paying
    forty times a query's cost to learn statistics for the next one. They are estimators
    fitted to a distribution, and a sample identifies that distribution as well as a census
    does: at 262,144 rows the sampling error sits below the sketches' own approximation
    error, so this loses nothing they could have told us. It is what `ANALYZE` does in every
    database that has one.

    The sample is **stratified by batch and taken as a random window within each**: every
    batch contributes in proportion to its size, so a column sorted across the input (a date,
    a key) is still covered end to end, and the window's random offset keeps a sorted batch
    from always donating its own low values. Both are zero-copy `slice`s — no row is read in
    Python, and the cost does not grow with the input.

    Args:
        batches: The scanned batches to sample.

    Returns:
        Zero-copy slices totalling at most `_SKETCH_SAMPLE_ROWS` rows — `batches` itself when
        it is already that small.
    """
    total = sum(b.num_rows for b in batches)
    if total <= _SKETCH_SAMPLE_ROWS:
        return batches
    fraction = _SKETCH_SAMPLE_ROWS / total
    out: list[pa.RecordBatch] = []
    for batch in batches:
        take = max(1, int(batch.num_rows * fraction))
        offset = _rng.randrange(batch.num_rows - take + 1) if batch.num_rows > take else 0
        out.append(batch.slice(offset, take))
    return out


def learn_column_stats(
    hub,
    resolved: list[list[pa.RecordBatch]],
    sources: list[Source] | None = None,
    plan: LogicalPlan | None = None,
    complete_scan: list[bool] | None = None,
) -> None:
    """Measure per-column ndv/quantiles from the just-scanned input and record them.

    `resolved[i]` are the batches scanned from `sources[i]`, and each source's statistics
    are recorded **under that source's identity** — because a column name alone does not
    identify a column, and an unqualified `{name: stat}` map lets one table's `id` answer
    for another's on every join and group-by estimate in the process. A source that cannot
    key itself is skipped rather than merged into the global namespace.

    Core measures (`core.column_statistics`); Kyber persists/consumes. Best-effort: a
    failure here never affects the query result.

    ## Only the columns something will read, and never unboundedly

    The sketch is an **O(rows x columns)** pass — HLL, KLL and Misra-Gries over every value —
    and it used to run over *every column of the source*, for every query, whether or not any
    of it could ever be consulted. On a plain ``read_parquet(dir).collect()`` — 20M rows, 16
    columns, no join, no group-by, no filter, and therefore not one statistic the estimator
    can use — it cost **22.9 seconds on top of a 0.73-second read**. The query paid 30x its
    own cost to learn things nothing would ever ask for.

    So it is bounded exactly the way the pre-optimize `seed_column_ndv` already bounds
    itself, and for the same stated reason: *computing a column the optimizer does not read
    only wastes work.*

    * `plan` restricts the sketch to `learnable_columns` — the join keys, group keys and
      filter columns an estimator actually consults. A filter-free, join-free scan learns
      nothing, because there is nothing to learn.
    * `ndv_sketch_max_cells` caps the total work, so one enormous column cannot turn a
      cheap query into an expensive one. The pre-pass has always honored this cap; this one
      did not, which is why it had no ceiling at all.
    * **The costly sketches read a sample, not the whole input** (`_sketch_sample`). Bounding
      the *columns* still left the pass reading every *row* of them: the KLL and Misra-Gries
      sketches cost ~56 ns a cell against HyperLogLog's ~4, so a cold 3 ms `filter`+`group_by`
      over 1M rows spent **114 ms** in this function — 38x the query, to inform the next one.
      Distinct counts still read every row (HLL is cheap, and distinct-count is the one
      statistic a sample genuinely cannot give you); quantiles, byte widths and
      most-common-values are fitted to a bounded sample, which is what `ANALYZE` does
      everywhere else and what their own error bars already assume.

    Passing `plan=None` keeps the old learn-everything behavior, for the caller that hands
    in an already-bounded *sample* (`collect_source_metadata`) rather than a whole scan.

    The "already measured" marker is the *average byte width*, not the distinct count:
    `column_statistics` records one for every column it touches (numeric or not), whereas
    `ndv` alone is also written by the cheaper pre-optimize `seed_column_ndv`. Gating on
    `ndv` would let that seeding suppress this pass, losing the quantile grids and
    most-common-values it is the only source of. A column left unsketched stays unmarked, so
    the first query that *can* use it is the one that measures it.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        learned = kyber.load_learned_stats(hub)
        min_frac = active_config().optimizer.cardinality.mcv_min_fraction
        max_cells = active_config().optimizer.ndv_sketch_max_cells
        wanted = learnable_columns(plan) if plan is not None else None
        # Byte widths first, for **every** column, and before the `wanted` gate returns. They
        # are not a sketch: Arrow already knows each array's buffer size, so this is
        # `nbytes / rows` per column and costs nothing worth measuring. It has to be outside
        # the gate because the gate is about *distribution* statistics — a column no predicate
        # mentions has no use for a quantile grid, and is very often the widest column in the
        # row. See `record_column_row_bytes`.
        _learn_row_bytes(hub, resolved, sources)
        if wanted is not None and not wanted:
            return  # nothing in this plan consults a column statistic
        for i, batches in enumerate(resolved):
            if not batches:
                continue
            source = sources[i] if sources is not None and i < len(sources) else None
            source_key = source_stats_key(source) if source is not None else None
            if source_key is None:
                continue  # unkeyable: cannot be told apart from another source's columns
            # An adaptive stage boundary's intermediate is keyed by object identity and does
            # not survive the execution, so everything sketched from it is written under a
            # key no later query can name. See `seed_column_ndv` for the three costs that
            # carries; this pass is the expensive one (KLL + Misra-Gries, ~56 ns a cell).
            if getattr(source, "ephemeral", False):
                continue
            known = set(kyber.columns_for(learned, kyber.AVG_BYTES_KEY, source_key))
            cols = [
                c
                for c in batches[0].schema.names
                if c not in known and (wanted is None or c in wanted)
            ]
            if not cols:
                continue
            rows = sum(b.num_rows for b in batches)
            if rows * len(cols) > max_cells:
                continue  # too big to sketch cheaply; a worse plan beats a 20x slower query
            # A distinct count is the one measured statistic a *partial* scan gets **wrong**,
            # not merely approximate. A quantile grid or an MCV from a sample is still a valid
            # description of the whole column's distribution — sampling preserves shape. But an
            # ndv from a subset of the rows counts only the distinct values in that subset: a
            # `filter(id < 100).collect()` that scanned 100 of a table's 2,000,000 rows would
            # record `ndv=100` under the *source's* key, and `approx_n_unique("id")` — which
            # reads exactly that record — would then answer 100 for the whole table. So the ndv
            # is recorded only when this query scanned the source *whole* (no predicate pushed,
            # every row read); a filtered or limited scan still contributes quantiles and MCVs.
            saw_whole = complete_scan is None or (i < len(complete_scan) and complete_scan[i])
            # Distinct counts read every row; the quantile/MCV sketches read a bounded
            # sample. See `_sketch_sample` — one is ~4 ns a cell, the others ~56.
            ndv = core.column_ndv(batches, cols) if saw_whole else {}
            sample = _sketch_sample(batches)
            total = sum(b.num_rows for b in sample)
            _sample_ndv, quants, avg_bytes = core.column_statistics(sample, cols)
            mcv: dict[str, dict[str, float]] = {}
            # Heavy hitters are measured on **every** column being sketched, not only
            # low-cardinality ones.
            #
            # This used to skip any column with `ndv > 1/min_frac` (20), reasoning that a
            # high-cardinality column cannot hold a value above the frequency floor. That is
            # only true under *uniformity* — which is the one assumption an MCV exists to
            # correct. A column of a million distinct keys can still have a single value at
            # 30% of rows (a sentinel, a default account, one whale customer), and
            # Misra-Gries finds it in the same pass. Measured: 1,000 distinct `cust_id`s with
            # key 7 at 47.5% of rows was excluded by the gate, so `cust_id = 7` estimated at
            # `1/ndv` = 0.001 against a true 0.5 — a ~500x under-estimate on the most skewed
            # key in the table, which is exactly the key a join is about to be built on. The
            # gate suppressed skew precisely where skew matters, leaving join-key skew
            # structurally unmeasurable (`kyber.hot_join_values` reads this).
            # Over the sample, and against the sample's row count: an MCV is a *fraction*
            # of rows, and a uniform sample preserves a value's frequency.
            for col_name, hits in core.heavy_hitters(sample, cols, min_frac).items():
                if total > 0 and hits:
                    mcv[col_name] = {str(v): n / total for v, n in hits}
            if ndv or quants or avg_bytes or mcv:
                kyber.record_column_stats(hub, ndv, quants, avg_bytes, mcv, source_key=source_key)
    except Exception:  # learning must never break execution — but it must not vanish either
        # This `except` is load-bearing (a measurement failure must never fail a query), but
        # a bare `pass` also swallows a *bug*: an `AttributeError` on this function's first
        # line silently made the entire post-run column learner a no-op, so quantiles, MCVs
        # and byte widths were never recorded at all. A swallowed exception that is never
        # logged is indistinguishable from a code path that works.
        _log.debug("column-statistics learning failed; plans fall back to priors", exc_info=True)
