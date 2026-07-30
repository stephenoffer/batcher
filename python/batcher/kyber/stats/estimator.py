"""`StatsEstimator` — propagate `RelStats` (rows + column stats) through a plan.

This is the metadata-first heart of Kyber's cost layer. For every operator it
estimates the output row count *and* per-column statistics, each tagged with a
`Provenance`. Base sizes and column bounds come from sources cheaply (a
`SourceStatistics` carries footer/manifest row counts and min/max);
selectivities and join-key distinct counts are refined across executions from
the MetadataHub (the learning loop). Crucially, a statistic is tagged `EXACT`
only when it is provably correct without execution — that is the gate the
metadata-answer layer (`count()`, `min()`, `is_empty()`, …) reads.

`estimate(node) -> RelStats` is the single entry point. Row logic lives here;
column-stat propagation is delegated to `columns`, predicate selectivity to
`selectivity`. The public name `CardinalityEstimator` is preserved as an alias
in `batcher.kyber.cardinality` for back-compat.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import pyarrow as pa

from batcher.config import CardinalityConfig, active_config
from batcher.kyber.column_tables import (
    AVG_BYTES_KEY,
    CARDINALITY_CORRECTION_KEY,
    MCV_KEY,
    NDV_KEY,
    QUANTILES_KEY,
    columns_for,
)
from batcher.kyber.properties import project_ordering
from batcher.kyber.stats import columns as col_prop
from batcher.kyber.stats.distribution import (
    join_match_fraction,
    mcv_join_rows,
    overlap_fraction,
    union_ndv,
)
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.kyber.stats.selectivity.scalars import _fraction_below_quantiles, _ordinal
from batcher.plan.expr_ir import Col, Expr, IsNotNull, Lit
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    RangeJoin,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    Window,
    is_cartesian_key_pair,
)
from batcher.plan.source_stats import SourceStatistics, source_stats_key
from batcher.plan.stats import ColumnStat, Provenance, RelStats, weakest
from batcher.plan.types import column_bytes

__all__ = ["StatsEstimator", "combine_ndv"]

# Selectivity of an inequality whose operand ranges are unknown. System R's constant, and
# the one DuckDB, Postgres and Spark all still use.
_UNKNOWN_INEQUALITY_SELECTIVITY = 1.0 / 3.0

# Floor for a computed inequality selectivity. Zero would assert the join is *empty*, which
# a distribution assumption may not do — only a proof may.
_MIN_INEQUALITY_SELECTIVITY = 1e-6


def _static_unnest_fanout(node: Unnest) -> float | None:
    """Rows per input row an `Unnest` produces, when the *type* proves it.

    A `fixed_size_list` carries its length, so exploding it is an exact fan-out — the same
    kind of data-independent multiplier `Unpivot` gets from its column count, and the same
    fact `plan.types.column_bytes` already reads to size the column. This is the embedding
    and fixed-shape-vector case, and it is common enough in AI pipelines that leaving it to
    the learning loop means being wrong by the vector's dimension on every cold run.

    `None` for a variable-length list, whose length genuinely is a property of the data.

    Args:
        node: The explode being estimated.

    Returns:
        The exact fan-out, or `None` when the type does not prove one.
    """
    schema = node.input.available_schema()
    if schema is None:
        return None
    field = next((f for f in schema.arrow if f.name == node.column), None)
    if field is None:
        return None
    dtype = field.type
    # An extension type is a label on a storage layout, and a fixed-shape tensor's storage
    # is exactly the fixed-size list this is looking for — the same unwrap `column_bytes`
    # needs, and for the same reason: no `pa.types.is_*` predicate sees through the label.
    storage = getattr(dtype, "storage_type", None)
    if isinstance(storage, pa.DataType):
        dtype = storage
    if not pa.types.is_fixed_size_list(dtype):
        return None
    # An `outer` explode keeps a row whose list is null, so it can never emit fewer rows
    # than it read; a zero-length fixed list would otherwise estimate the relation empty.
    return float(max(1, dtype.list_size)) if node.outer else float(dtype.list_size)


def _uniform_p_less(a1: float, b1: float, a2: float, b2: float) -> float:
    """`P(X < Y)` for independent `X ~ U[a1, b1]` and `Y ~ U[a2, b2]`.

    Integrating `X`'s CDF over `Y`'s support: the part of `Y`'s range above `b1` contributes
    1, the part below `a1` contributes 0, and the overlap contributes the area under the
    ramp between them. Degenerate (zero-width) ranges are the point-mass limits of the same
    expression and are handled explicitly rather than by dividing by zero.
    """
    if b1 <= a2:
        return 1.0
    if a1 >= b2:
        return 0.0
    if b1 == a1:  # X is a point
        return 0.0 if b2 == a2 else min(1.0, max(0.0, (b2 - a1) / (b2 - a2)))
    if b2 == a2:  # Y is a point
        return min(1.0, max(0.0, (a2 - a1) / (b1 - a1)))
    above = max(0.0, b2 - max(a2, b1))
    lo, hi = max(a2, a1), min(b2, b1)
    ramp = ((hi - a1) ** 2 - (lo - a1) ** 2) / (2.0 * (b1 - a1)) if hi > lo else 0.0
    return min(1.0, max(0.0, (above + ramp) / (b2 - a2)))


# Operators whose estimates carry a learned correction. Joins, aggregates, and distincts
# are where structural cardinality estimation is worst (containment assumptions, ndv
# products) and where being wrong is most expensive — a mis-sized build side or shuffle.
# `Unnest` belongs for a different reason: its fan-out is the average list length, which
# is a property of the *data*, not the plan — no structural rule can know it, so the
# estimator passes the child count through and is wrong by exactly that factor (a RAG
# pipeline chunking documents 20-to-1 under-sizes every stage below it). Measuring it
# once and correcting is the only way, and is precisely the Core-measures/Kyber-consumes
# loop.
# `MapBatches` belongs for the same reason as `Unnest`: a UDF may filter, explode, or pass
# rows through 1:1, and which one is a property of the *code*, not the plan — the structural
# estimator can only assume 1:1 and is wrong by the true fan-out on the first run. A
# filtering embedding-dedup or an exploding chunker (both common AI-pipeline stages) is then
# mis-sized for every stage below it until the measured ratio corrects it. This is safe only
# because the `map_batches` signature now carries the UDF's identity (see
# `kyber.signature._udf_identity`), so one UDF's learned fan-out cannot answer for another's.
# `Filter` is deliberately excluded: its measured *selectivity* is already learned
# per-signature, and correcting it here as well would count the same error twice.
# Row-preserving operators (Project, Sort, Limit) need no correction — they inherit it
# from their input.
_CORRECTABLE = (Aggregate, Distinct, Join, MapBatches, Unnest)


class StatsEstimator:
    """Estimates per-operator output cardinality and column statistics.

    `sources` are the bound inputs (indexed by a `Scan`'s `source_id`);
    `source_stats` is an optional parallel list of `SourceStatistics` the
    conductor collected at plan-build time (footer/manifest stats), used to seed
    exact base-relation row counts and column bounds. `learned` is the
    MetadataHub blob (per-signature cardinality, `__column_ndv__`, quantiles,
    byte widths).
    """

    def __init__(
        self,
        sources: list,
        learned: dict[str, Any] | None = None,
        cfg: CardinalityConfig | None = None,
        source_stats: list[SourceStatistics | None] | None = None,
        exact_first: bool = False,
    ) -> None:
        self._sources = sources
        # learned[signature] -> {"selectivity": float, "rows": float}
        self._learned = learned or {}
        self._cfg = cfg or active_config().optimizer.cardinality
        self._source_stats = source_stats
        # When True, a learned absolute row count never shadows an exact structural
        # estimate — the metadata-answer path needs EXACT to win over LEARNED so a
        # provably-correct count isn't masked by a (also-correct but weaker-tagged)
        # measurement from a past run. Learned ndv/selectivity still apply.
        self._exact_first = exact_first
        # Per-run memo caches keyed by node identity. The plan is immutable for this
        # estimator's lifetime, so `estimate(node)` and a node's structural signature
        # are pure functions of the node. Without memoization `estimate` re-descends
        # to the leaves on every call and is invoked O(nodes) times per optimize (once
        # per node in `annotate_ops`, plus every cost-based rule), so planning is
        # super-linear in plan depth and `plan_signature` re-hashes whole subtrees.
        # Each entry holds a strong reference to its keyed node alongside the value so
        # a freed node's reused `id()` can never produce a stale hit.
        self._row_cache: dict[int, tuple[LogicalPlan, RelStats]] = {}
        self._sig_cache: dict[int, tuple[LogicalPlan, str]] = {}
        # Per-source learned column stats (`{source_id: {column: ColumnStat}}`), built
        # lazily per source. Resolving them **per source** is the whole point: the learned
        # maps are keyed by `(source, column)`, so a `Scan` gets its *own* table's measured
        # ndv/quantiles/mcv/width and never another table's column of the same name.
        self._learned_cols: dict[int, dict[str, ColumnStat]] = {}

    def estimate(self, node: LogicalPlan) -> RelStats:
        """Cardinality + column stats for `node`, memoized by node identity for the
        duration of this estimator (one optimize run)."""
        cached = self._row_cache.get(id(node))
        if cached is not None and cached[0] is node:
            return cached[1]
        result = self._corrected(node, self._estimate_uncached(node))
        self._row_cache[id(node)] = (node, result)
        return result

    def signature_of(self, node: LogicalPlan) -> str:
        """The node's structural plan signature — its identity across executions.

        Memoized for this run. Public so the plan annotator can stamp it onto each
        `PhysicalOp`, giving Core a stable key to report measured cardinality against
        (`op_id` is only a position in this plan's walk).

        Args:
            node: The plan node to identify.

        Returns:
            The structural signature string.
        """
        return self._sig(node)

    def correction_for(self, node: LogicalPlan) -> float:
        """The learned cardinality-correction factor `estimate` applies to `node`.

        1.0 when nothing is learned or the operator is not correctable.

        Args:
            node: The plan node being estimated.

        Returns:
            The multiplicative factor applied to the structural row estimate.
        """
        if not isinstance(node, _CORRECTABLE):
            return 1.0
        if self._learned_rows_win(node):
            return 1.0  # a measured absolute size supersedes any correction
        factor = self._corrections.get(self._sig(node))
        if factor is None or factor <= 0.0:
            return 1.0
        return float(factor)

    def reportable_estimate(self, node: LogicalPlan) -> float:
        """The row estimate to report as feedback, or `0.0` to report nothing.

        The correction loop learns how wrong the **structural** estimator is, so a sample
        is only meaningful when this run's estimate for `node` actually came from that
        estimator. Three cases teach it nothing and must not become samples:

        * a measured absolute row count (`_learned_rows_win`) — the estimate is then a
          past measurement, so its q-error is ~1.0 by construction. Averaging those in
          would drag a hard-won correction back toward 1.0 as a query is re-run.
        * an `EXACT` estimate — provably right, nothing to correct.
        * the `unknown_rows` placeholder — not an estimate at all.

        Otherwise the correction this run applied is divided back out, yielding the raw
        structural estimate whose error the loop is measuring.

        Args:
            node: The plan node being estimated.

        Returns:
            The pre-correction structural row estimate, or 0.0 when it teaches nothing.
        """
        if not isinstance(node, _CORRECTABLE) or self._learned_rows_win(node):
            return 0.0
        est = self.estimate(node)
        if est.provenance is Provenance.EXACT or est.rows >= self._cfg.unknown_rows:
            return 0.0
        return est.rows / self.correction_for(node)

    def _learned_rows_win(self, node: LogicalPlan) -> bool:
        """Whether `_estimate_uncached` short-circuits to a measured absolute row count."""
        if self._exact_first or isinstance(node, Filter):
            return False
        learned = self._learned.get(self._sig(node))
        return learned is not None and "rows" in learned

    @property
    def _corrections(self) -> dict[str, float]:
        """Learned per-signature cardinality corrections (`{signature: factor}`).

        Derived by `kyber.learning` from the measured q-error history; empty until a
        correctable operator has run often enough to be trusted.
        """
        return self._learned.get(CARDINALITY_CORRECTION_KEY, {})

    def _corrected(self, node: LogicalPlan, stats: RelStats) -> RelStats:
        """Scale a structural estimate by what past executions measured it to be wrong by.

        This is the loop DuckDB, Polars, and Daft do not have, and that Spark AQE closes
        only *within* one query: an operator whose output Kyber has historically
        under-estimated 8x gets its next estimate multiplied by 8. Never touches an
        `EXACT` estimate (a provable count needs no correction, and must not be
        downgraded), never touches the `unknown_rows` placeholder (which is not an
        estimate at all), and always downgrades provenance to at best `LEARNED`.
        """
        if stats.provenance is Provenance.EXACT or stats.rows >= self._cfg.unknown_rows:
            return stats
        factor = self.correction_for(node)
        if factor == 1.0:
            return stats
        return replace(
            stats,
            rows=max(1.0, stats.rows * factor),
            provenance=weakest(stats.provenance, Provenance.LEARNED),
        )

    def _sig(self, node: LogicalPlan) -> str:
        """The node's structural signature, memoized by identity (see `estimate`)."""
        cached = self._sig_cache.get(id(node))
        if cached is not None and cached[0] is node:
            return cached[1]
        sig = _signature(node)
        self._sig_cache[id(node)] = (node, sig)
        return sig

    def _estimate_uncached(self, node: LogicalPlan) -> RelStats:
        # Learned-first: trust a measured absolute size for this exact shape — but only for
        # the operators a learned row count is *about*. `_CORRECTABLE` is that set, and the
        # rest are excluded for reasons that all bite here:
        #
        #   - A `Scan`'s cardinality is not a thing to learn. The source already reports it
        #     EXACTLY (a Parquet footer, an in-memory row count), and a learned value can
        #     only shadow that exact number with a weaker one.
        #   - Worse, `plan_signature` structures every scan as the bare token ``["scan"]``,
        #     carrying no source identity — so *all* scans in a process share one learned
        #     entry. Reading a 5M-row table therefore taught the optimizer that a 1,000-row
        #     change set also has 5M rows, which made a pruned MERGE size its join at 2.4 TB
        #     and spill a 100,000-row build side to disk. One table's measurement must never
        #     become another table's estimate.
        #   - A row-preserving operator (Project/Sort/Limit) inherits its input's count, and
        #     a `Filter`'s learned *selectivity* ratio (applied below to the current input)
        #     generalizes across input sizes better than a stale absolute count.
        if not self._exact_first and isinstance(node, _CORRECTABLE):
            learned = self._learned.get(self._sig(node))
            if learned is not None and "rows" in learned:
                return RelStats(float(learned["rows"]), Provenance.LEARNED)

        if isinstance(node, Scan):
            return self._estimate_scan(node)
        if isinstance(node, Filter):
            return self._estimate_filter(node)
        if isinstance(node, Project):
            child = self.estimate(node.input)
            columns = col_prop.project_columns(node.items, child, node.input.available_schema())
            # A projection reorders nothing, so the input's row order survives under the
            # output names (`kyber.properties.project_ordering`). Dropping it here — which is
            # what this did — lost the delivered order at exactly the node that sits between
            # a sort and its consumer in every real query, so the redundant-sort rule could
            # never see across a `SELECT`.
            ordering = project_ordering(node.items, child.sorted_by)
            return RelStats(child.rows, child.provenance, columns, ordering)
        if isinstance(node, MapBatches):
            # A UDF may filter/explode/pass-through — a property of the code the structural
            # estimator can't see, so it assumes 1:1 and lets the measured fan-out from a
            # previous run correct it (`MapBatches` is `_CORRECTABLE`, keyed by UDF identity).
            # The opaque UDF means output columns are unknown.
            return RelStats(self.estimate(node.input).rows, Provenance.DEFAULT)
        if isinstance(node, Unnest):
            # Explode multiplies rows by the average list length. For a *variable*-length
            # list that is a property of the data which no structural rule can know, so the
            # neutral (1x) default is wrong by exactly that factor on the first run;
            # `Unnest` is `_CORRECTABLE`, so a measured fan-out from a previous run is
            # applied by `estimate` on top of this.
            #
            # For a **fixed-size list** it is not unknown at all — the length is in the type,
            # exactly as it is for `Unpivot`'s column count one branch down, and exactly as
            # `plan.types.column_bytes` reads it to size the column's bytes. That is the
            # embedding/vector column: exploding a `fixed_size_list<float32, 768>` produces
            # 768 rows per input row, and estimating it at 1x under-sized every stage below
            # it by nearly three orders of magnitude on the first run — the run with nothing
            # learned, and the one that has to be admitted against a real envelope.
            child = self.estimate(node.input)
            fanout = _static_unnest_fanout(node)
            rows = child.rows * fanout if fanout is not None else child.rows
            # A proven fan-out is as exact as its input; a defaulted one is a guess.
            prov = child.provenance if fanout is not None else Provenance.DEFAULT
            return RelStats(rows, prov, col_prop.unnest_columns(node, child))
        if isinstance(node, Unpivot):
            # Unpivot emits one row per `on` column — an exact, data-independent fan-out.
            child = self.estimate(node.input)
            rows = child.rows * max(1, len(node.on))
            return RelStats(rows, child.provenance, col_prop.unpivot_columns(node, child))
        if isinstance(node, Sample):
            child = self.estimate(node.input)
            # A fixed-count sample yields exactly min(n, input) — a known small bound even
            # over an uncountable source, so it is a real estimate.
            if node.n is not None:
                rows = min(child.rows, float(node.n))
            # A *fractional* sample of an unknown-size input is itself unknown: scaling the
            # placeholder down (× fraction) would drop it below the "unknown" threshold and
            # let admission treat a guess as a budgetable estimate — the same trap the
            # aggregate/distinct estimators guard (a small query wrongly rejected as
            # infeasible). Keep it a placeholder; only scale a genuinely-known count.
            elif child.rows >= self._cfg.unknown_rows:
                rows = child.rows
            else:
                rows = child.rows * node.fraction
            return RelStats(rows, Provenance.DEFAULT, col_prop.sample_columns(child, rows))
        if isinstance(node, RowId):
            # `with_row_index` is strictly 1:1 — it appends a counter and changes nothing
            # else — so rows, provenance, column stats and the delivered ordering all carry
            # through. Falling to the `unknown_rows` default (which is what a missing branch
            # here does) turned an EXACT 1,000-row relation into a 1e12-row guess for every
            # operator above it: a join above a `with_row_index` picked the wrong build side
            # and admission sized the query against a fiction.
            child = self.estimate(node.input)
            return RelStats(
                child.rows,
                child.provenance,
                col_prop.row_id_columns(node, child),
                # The counter is ascending and never null, so it is a valid ordering in its
                # own right — recorded only when the child delivers none, since a data-column
                # ordering is what a downstream `Sort` is far more likely to be elided by.
                child.sorted_by or (node.alias,),
            )
        if isinstance(node, Aggregate):
            return self._estimate_aggregate(node)
        if isinstance(node, Sort):
            return self._estimate_sort(node)
        if isinstance(node, Window):
            return self._estimate_window(node)
        if isinstance(node, Limit):
            return self._estimate_limit(node)
        if isinstance(node, Distinct):
            return self._estimate_distinct(node)
        if isinstance(node, Union):
            return self._estimate_union(node)
        if isinstance(node, Join):
            return self._estimate_join(node)
        if isinstance(node, RangeJoin):
            return self._estimate_range_join(node)
        if isinstance(node, AsofJoin):
            # ASOF is left-style: exactly one output row per left row, so the count
            # (and its provenance) is the left input's — EXACT when the left is, so
            # `asof_join(...).count()` answers from metadata (incl. an empty left → 0).
            # The left columns are preserved 1:1 and the right survive as bounds, so they
            # propagate too instead of blinding every operator above the join.
            left = self.estimate(node.left)
            right = self.estimate(node.right)
            return RelStats(
                left.rows, left.provenance, col_prop.asof_join_columns(node, left, right)
            )
        return RelStats(self._cfg.unknown_rows, Provenance.DEFAULT)

    # --- per-operator estimators ------------------------------------------
    def _estimate_scan(self, node: Scan) -> RelStats:
        # The `Scan` leaf is where measured statistics enter the plan, and where they are
        # bound to the source that was actually measured. Everything above reads them off
        # `RelStats.columns` as they propagate.
        learned = self.learned_columns(node.source_id)
        src_stats = self._stats_for(node.source_id)
        if src_stats is not None:
            base = src_stats.to_relstats(default_rows=self._cfg.unknown_rows)
            columns = col_prop.scan_columns(base.columns, learned)
            return RelStats(base.rows, base.provenance, columns, base.sorted_by)
        # Sources may be absent (plan-shape optimization with no bound inputs) or
        # duck-typed without `row_count`; treat either as unknown rather than crash.
        source = self._sources[node.source_id] if node.source_id < len(self._sources) else None
        row_count_fn = getattr(source, "row_count", None)
        n = row_count_fn() if callable(row_count_fn) else None
        columns = col_prop.scan_columns({}, learned)
        if n is None:
            return RelStats(self._cfg.unknown_rows, Provenance.DEFAULT, columns)
        return RelStats(float(n), Provenance.EXACT, columns)

    def _estimate_filter(self, node: Filter) -> RelStats:
        child = self.estimate(node.input)
        # A constant-boolean predicate is provable without touching a row: `filter(TRUE)`
        # keeps the child exactly (rows + all column stats, EXACT preserved), and
        # `filter(FALSE)` is provably empty — an EXACT zero, the canonical empty marker,
        # letting `count()`/`is_empty()` answer a contradiction-filtered subtree from
        # metadata even over an otherwise-unknown source.
        if isinstance(node.predicate, Lit) and isinstance(node.predicate.value, bool):
            if node.predicate.value:
                return child
            return RelStats(0.0, Provenance.EXACT)
        # A filter over a **provably empty** relation is provably empty, whatever the
        # predicate: it keeps a subset of no rows. Without this, any filter placed above an
        # empty subtree downgrades it to a mere estimate and the emptiness proof is lost —
        # which is not hypothetical, because the optimizer itself *inserts* filters there
        # (a runtime/sideways join filter lands on the join's inputs). An inner join with an
        # EXACT-empty side then stopped answering `count() == 0` from metadata and executed
        # the join instead. Emptiness is the one property a filter can never destroy.
        if child.rows == 0 and child.provenance is Provenance.EXACT:
            return RelStats(0.0, Provenance.EXACT)
        # `col IS NOT NULL` is the one predicate whose effect is *recorded*: it drops exactly
        # the rows the column's null count counts. So the surviving row count is EXACT, and so
        # are that column's bounds — `min`/`max`/`ndv` are defined over the non-null values, and
        # dropping the nulls removes none of them.
        #
        # This is not a corner case, it is the common one: the optimizer *inserts* these filters
        # itself, one on each side of every equi-join (`push_is_not_null_from_join_key`). Losing
        # exactness here meant one rule destroyed precisely the statistic another needed, and a
        # join whose key ranges provably cannot overlap ran a full shuffle to discover what the
        # two footers already said.
        exact = self._not_null_stats(node, child)
        if exact is not None:
            return exact
        sel = self._selectivity(node, child)
        # `prov` is LEARNED (measured selectivity) or DEFAULT (Selinger) — never
        # EXACT — so a filtered row count is never EXACT, however exact the child.
        prov = Provenance.LEARNED if self._has_learned(node) else Provenance.DEFAULT
        out_rows = child.rows * sel
        return RelStats(
            out_rows,
            weakest(child.provenance, prov),
            col_prop.filter_columns(child, out_rows),
            child.sorted_by,
        )

    def _not_null_stats(self, node: Filter, child: RelStats) -> RelStats | None:
        """EXACT stats for a `Filter(col IS NOT NULL)`, or None when this isn't that shape.

        The one filter whose effect is already *recorded*: it drops exactly the rows the
        column's null count counts. So the surviving count is EXACT — and so are that column's
        bounds, because `min`/`max`/`ndv` are defined over the non-null values and dropping the
        nulls removes none of them. Its null count becomes a known zero.

        Every **other** column still downgrades, and must: a row dropped for a null in `col` may
        have carried the extreme value of some other column.

        Needs an EXACT child row count and an EXACT null count for the tested column; any weaker
        input falls through to the ordinary selectivity estimate, exactly as before.
        """
        pred = node.predicate
        if not isinstance(pred, IsNotNull) or not isinstance(pred.input, Col):
            return None
        name = pred.input.name
        stat = child.columns.get(name)
        if (
            not child.rows_exact
            or stat is None
            or stat.null_count is None
            or not stat.null_count_is_exact
        ):
            return None
        surviving = float(child.rows - stat.null_count)
        columns = col_prop.filter_columns(child, surviving)
        # The tested column's own statistics survive the filter (see the docstring); its null
        # count is now a known zero and its ndv is unchanged (dropping nulls removes no distinct
        # value), so restore both after the generic downgrade. Its *bundle* provenance is
        # whatever it was — a string column keeps its truncatable bounds untrusted, and that is
        # right.
        columns[name] = replace(stat, null_count=0.0)
        return RelStats(surviving, Provenance.EXACT, columns, child.sorted_by)

    def _estimate_sort(self, node: Sort) -> RelStats:
        child = self.estimate(node.input)
        if node.limit is not None:
            # A top-N (fused Sort+Limit) keeps only `limit` rows and can exclude a column's
            # extremes, so its `min`/`max`/`ndv` must downgrade to bounds exactly as a
            # `Filter`/`Limit` does — leaving them EXACT would let `min()`/`count_distinct()`
            # answer from metadata over rows the top-N dropped.
            rows = min(child.rows, float(node.limit))
            return RelStats(
                rows,
                child.provenance,
                col_prop.limit_columns(child, rows),
                _canonical_sort_prefix(node.keys),
            )
        # A full sort preserves the exact value set, so column stats pass through unchanged.
        return RelStats(
            child.rows, child.provenance, dict(child.columns), _canonical_sort_prefix(node.keys)
        )

    def _estimate_limit(self, node: Limit) -> RelStats:
        child = self.estimate(node.input)
        rows = min(child.rows, float(node.n))
        # `Limit(x, 0)` is provably empty regardless of the child's provenance (it is
        # the canonical empty marker), so its zero row count is EXACT — letting
        # `count()`/`is_empty()` answer a pruned-to-empty subtree from metadata even
        # over an unknown source. Otherwise the (possibly truncated) count is as exact
        # as the child.
        prov = Provenance.EXACT if node.n == 0 else child.provenance
        return RelStats(rows, prov, col_prop.limit_columns(child, rows), child.sorted_by)

    def _estimate_union(self, node: Union) -> RelStats:
        children = [self.estimate(i) for i in node.inputs]
        # A union of provably-empty branches is provably empty (concatenation invents no
        # row), so `count()`/`is_empty()` answer 0 from metadata — the same emptiness proof
        # a filter/join preserves. Without this an all-pruned union executed to discover 0.
        if children and all(c.rows_exact and c.rows == 0 for c in children):
            return RelStats(0.0, Provenance.EXACT)
        total = sum(c.rows for c in children)
        prov = weakest(*(c.provenance for c in children)) if children else Provenance.DEFAULT
        names = node.available_columns()
        columns = col_prop.union_columns(children, names)
        if node.distinct:
            # `UNION` (not `UNION ALL`) dedups across branches, so its output is the distinct
            # count of the concatenation — bounded below by the largest branch and above by
            # `Σ n_i`, never above. Reporting the un-deduplicated `total` was an upper bound
            # only, and it made a `UNION` of near-identical partitions look like it doubled the
            # data. `union_ndv` interpolates between those bounds with the same
            # independent-membership model the column-stat merge uses, so the row estimate and
            # the propagated per-column ndv cannot disagree.
            deduped = union_ndv([c.rows for c in children], total)
            largest = max(c.rows for c in children)
            rows = total if deduped is None else min(total, max(deduped, largest))
            return RelStats(rows, weakest(prov, Provenance.DEFAULT), columns)
        return RelStats(total, prov, columns)

    def _estimate_window(self, node: Window) -> RelStats:
        """A window appends columns — unless it is rank-limited, in which case it drops rows.

        `rank_limit` is the fused form of ``Filter(Window([row_number]), rn <= k)``
        (`kyber.rules.fusion`): the engine keeps only the top-`k` rows **per partition** and
        the `Filter` disappears from the plan. Treating that as row-preserving is not a soft
        mis-estimate, it is a wrong answer — the count carried the child's `EXACT`
        provenance, so `count()` answered it from metadata *without executing*, and
        ``distinct(subset=…, keep="last", order_by=…)`` (which lowers to exactly this shape)
        reported the number of input rows instead of the number of surviving ones. A merge
        whose source was deduplicated that way then failed its own cardinality check.

        Bounded above by `rank_limit` rows per partition, and never `EXACT`: a partition
        holding fewer than `k` rows contributes fewer, so the bound is not the count.
        """
        child = self.estimate(node.input)
        columns = col_prop.window_columns(node, child)
        if node.rank_limit is None:
            # Row-preserving: appends columns, never changes the row count, so the input
            # columns' stats (EXACT included) carry through untouched.
            return RelStats(child.rows, child.provenance, columns, child.sorted_by)

        partitions = self._partition_count(node, child)
        rows = min(child.rows, partitions * float(node.rank_limit))
        return RelStats(rows, Provenance.DEFAULT, columns, child.sorted_by)

    def _partition_count(self, node: Window, child: RelStats) -> float:
        """How many partitions the window's keys cut the input into (1 when unpartitioned).

        This is the distinct-combination count of the partition-key *set* — the same quantity
        a group-by, a `DISTINCT`, and a join key set all ask for — so it goes through the same
        damped combiner. Multiplying the per-key distinct counts (as this did) assumes the keys
        are independent, which the keys of a real `PARTITION BY (region, store)` never are; the
        product then saturates the row-count cap and a `QUALIFY rank <= k` is estimated to keep
        every row, defeating the rank-limit fusion it exists to size.
        """
        if not node.partition_keys:
            return 1.0
        ndv = _ndvs(child)
        per_key = []
        for key in node.partition_keys:
            if isinstance(key, Col) and key.name in ndv and ndv[key.name] > 0:
                per_key.append(ndv[key.name])
            else:
                return child.rows  # unknown ndv → assume every row its own partition
        return combine_ndv(per_key, child.rows)

    def _estimate_aggregate(self, node: Aggregate) -> RelStats:
        """Group-by output ≈ distinct group-key combinations; a global aggregate
        is exactly one row, with per-aggregate output values derived from the
        child's exact column stats when possible."""
        child = self.estimate(node.input)
        if not node.group_keys:
            columns = col_prop.global_aggregate_columns(node, child)
            return RelStats(1.0, Provenance.EXACT, columns)  # global aggregate → one row
        # A bare-`Col` group key carries its column's EXACT min/max forward as bounds
        # (grouping selects the distinct values, so the extremes are unchanged).
        key_cols = col_prop.grouped_aggregate_columns(node, child)
        if not self._exact_first:
            learned_rows = self._learned.get(self._sig(node), {}).get("rows")
            if learned_rows is not None:
                return RelStats(float(learned_rows), Provenance.LEARNED, key_cols)
        ndv = _ndvs(child)
        key_ndvs = [
            ndv[k.expr.name]
            for k in node.group_keys
            if isinstance(k.expr, Col) and k.expr.name in ndv and ndv[k.expr.name] > 0
        ]
        if len(key_ndvs) == len(node.group_keys):
            # Every key measured: the distinct combinations of the group-key set — the same
            # quantity a join computes for its key set, so the same (damped) combiner.
            # Multiplying the per-key counts assumed independence; correlated keys then
            # saturated the cap and the optimizer concluded that grouping reduced nothing.
            return RelStats(combine_ndv(key_ndvs, child.rows), Provenance.LEARNED, key_cols)
        # Not every key is measured. An unknown-placeholder input (an uncountable source —
        # `from_batches`, a stream, an un-pushed SQL scan) must NOT be shrunk below the
        # "unknown" threshold: the shrunk guess (0.1·unknown) is small enough to look like a
        # real estimate, so the optimizer would *budget* it and Carbonite could wrongly reject
        # an actually-small query. Keep it a placeholder (a guess never fails a real query).
        if child.rows >= self._cfg.unknown_rows:
            return RelStats(child.rows, Provenance.DEFAULT, key_cols)
        # Otherwise the blunt 0.1 fallback, but floored by the distinct combinations the
        # *measured* keys already imply — adding the unmeasured keys can only add groups, so
        # the known combination is a firm lower bound and the floor only raises the estimate
        # (over-budgeting the group hash table, the safe direction).
        estimate = max(1.0, child.rows * 0.1)
        if key_ndvs:
            estimate = max(estimate, combine_ndv(key_ndvs, child.rows))
        return RelStats(estimate, Provenance.DEFAULT, key_cols)

    def _estimate_distinct(self, node: Distinct) -> RelStats:
        """Dedup count ≈ the distinct combinations of the projected columns.

        The same quantity `Aggregate` estimates for its group keys, so the same
        `combine_ndv` combiner. For the common single-column `DISTINCT col` this is the
        column's measured ndv (~exact); a multi-column set is damped rather than
        multiplied, since the columns of a real key set are correlated. Falls back to 50%
        when any column's ndv is unmeasured."""
        child = self.estimate(node.input)
        cols = node.available_columns()
        ndv = _ndvs(child)
        columns = col_prop.distinct_columns(child)
        if cols and all(c in ndv and ndv[c] > 0 for c in cols):
            groups = combine_ndv((ndv[c] for c in cols), child.rows)
            return RelStats(groups, Provenance.LEARNED, columns)
        # Unknown-placeholder input → keep the placeholder (see `_estimate_aggregate`):
        # shrinking it would let admission wrongly reject a small query.
        if child.rows >= self._cfg.unknown_rows:
            return RelStats(child.rows, Provenance.DEFAULT, columns)
        # The blunt 50% fallback, floored by the distinct combinations the *measured* columns
        # already imply — a subset of the key columns is a firm lower bound on the full set's
        # distinct count, so the floor only raises the estimate (the safe, over-budget way).
        estimate = max(1.0, child.rows * 0.5)
        measured = [ndv[c] for c in cols if c in ndv and ndv[c] > 0]
        if measured:
            estimate = max(estimate, combine_ndv(measured, child.rows))
        return RelStats(estimate, Provenance.DEFAULT, columns)

    def _estimate_range_join(self, node: RangeJoin) -> RelStats:
        """`|L| x |R| x prod(selectivity_i)` — the product a cartesian join would produce,
        cut by each inequality's estimated selectivity.

        The alternative (falling through to the unknown-rows default) reported a fixed
        `1e12` for every range join, which is worse than useless: it is not merely
        imprecise, it is the same number regardless of input size, so join ordering and
        memory sizing above the operator could not tell a ten-row range join from a
        ten-million-row one.

        Conditions are combined by independence. That is optimistic for the shape the
        operator exists for — interval containment, where `lo` and `hi` are strongly
        correlated — so the estimate runs high there. It is the standard assumption and
        the direction that errs toward over-provisioning rather than under.
        """
        left = self.estimate(node.left)
        right = self.estimate(node.right)
        selectivity = 1.0
        for cond in node.conditions:
            selectivity *= self._inequality_selectivity(left, right, cond)
        rows = max(0.0, left.rows * right.rows * selectivity)
        columns = col_prop.range_join_columns(node, left, right, rows)
        return RelStats(rows, Provenance.DEFAULT, columns)

    def _inequality_selectivity(self, left: RelStats, right: RelStats, cond) -> float:
        """`P(left_key OP right_key)` for two independent uniform columns.

        With both `[min, max]` ranges known this is a closed form rather than a constant,
        which is what lets the estimator see that a join whose ranges barely overlap
        produces few rows and one whose ranges nest produces nearly the full product.
        Falls back to System R's 1/3 — the same constant DuckDB, Postgres and Spark use
        — when either range is missing or is not linearly ordered.
        """
        ls = left.columns.get(cond.left_key)
        rs = right.columns.get(cond.right_key)
        if ls is None or rs is None:
            return _UNKNOWN_INEQUALITY_SELECTIVITY
        bounds = [_ordinal(v) for v in (ls.min, ls.max, rs.min, rs.max)]
        if any(b is None for b in bounds):
            return _UNKNOWN_INEQUALITY_SELECTIVITY
        a1, b1, a2, b2 = bounds  # type: ignore[misc]
        if b1 < a1 or b2 < a2:
            return _UNKNOWN_INEQUALITY_SELECTIVITY
        p_less = _uniform_p_less(a1, b1, a2, b2)
        p = p_less if cond.op in ("lt", "le") else 1.0 - p_less
        # Never return exactly 0: a zero estimate is a *proof* of emptiness and this is an
        # assumption, so it must not let a downstream rule delete the join.
        return min(1.0, max(_MIN_INEQUALITY_SELECTIVITY, p))

    def _estimate_join(self, node: Join) -> RelStats:
        left = self.estimate(node.left)
        right = self.estimate(node.right)
        # Provable emptiness wins over any learned/Selinger estimate: when the
        # relevant side(s) are EXACT-empty the join is EXACT-empty too, so
        # `count()`/`is_empty()` answer 0 from metadata without executing the join.
        if _join_provably_empty(node.join_type, left, right):
            return RelStats(0.0, Provenance.EXACT)
        # A key pair whose `[min, max]` ranges do not overlap can share no value, so an
        # inner/semi join over it produces nothing — the join analogue of an out-of-bounds
        # equality. Kept a DEFAULT-provenance estimate (not an EXACT-empty *proof*): it
        # steers cost and join order toward killing the pipeline early without letting
        # `count()` answer 0 from metadata, so a mis-propagated bound can never become a
        # wrong result. Catches a filter-narrowed time-partition join (`WHERE d >= '2024'`
        # over a table ending in 2023) that structural containment estimates at full size.
        if node.join_type in ("inner", "semi") and self._join_keys_range_disjoint(
            node, left, right
        ):
            return RelStats(0.0, Provenance.DEFAULT, col_prop.join_columns(node, left, right, 0.0))
        rows, provenance = self._join_rows(node, left, right)
        # A preserved column's values carry through as downgraded *bounds* (a join
        # removes/duplicates rows but invents no value); never EXACT. The output row
        # count caps each carried-forward `ndv`, so a join above a join still knows its
        # key distinct counts (see `col_prop.join_columns`).
        columns = col_prop.join_columns(node, left, right, rows)
        return RelStats(rows, provenance, columns)

    def _join_rows(self, node: Join, left: RelStats, right: RelStats) -> tuple[float, Provenance]:
        """The join's estimated output cardinality and the provenance of that estimate.

        Each join type is derived from the *inner* estimate rather than sharing it:

        * `inner` — the Selinger containment estimate (`_inner_join_rows`).
        * `semi` — the left rows whose key matches: ``|L| x min(1, d_R/d_L)``.
        * `anti` — the complement, ``|L| - |semi|``. Semi and anti *partition* `|L|`, so
          returning `|L|` for both (as this did) is impossible unless one is empty; it
          costed a near-empty anti-join at full width.
        * `left`/`right`/`full` — an outer join **preserves** its outer side, so its output
          can never fall below that side's row count. Without this floor, a selective
          `LEFT JOIN` estimated below `|L|` — a count no execution can produce.
        * a cartesian pseudo-join — ``|L| x |R|``, not ``max(|L|, |R|)``.
        """
        if not self._exact_first:
            learned_rows = self._learned.get(self._sig(node), {}).get("rows")
            if learned_rows is not None:
                return float(learned_rows), Provenance.LEARNED
        if self._is_cartesian(node):
            return self._cartesian_rows(node, left, right)

        left_ndv = self._side_ndv(node.left_keys, left)
        right_ndv = self._side_ndv(node.right_keys, right)

        if node.join_type in ("semi", "anti"):
            return self._semi_anti_rows(node, left, right, left_ndv, right_ndv)

        inner = self._inner_join_rows(node, left, right, left_ndv, right_ndv)
        if node.join_type == "inner":
            return inner, Provenance.DEFAULT
        # An outer join emits the inner result **plus** a null-extended row for each unmatched
        # row of its preserved side. Both terms are needed: `max(inner, |L|)` (the previous
        # form) is only a lower bound, and it collapses to exactly `|L|` for the shape that
        # matters most — a `LEFT JOIN` to a dimension that fans out — where the true size is
        # `inner + unmatched` and can be several times larger. Under-estimating an outer join
        # is the direction that under-sizes its hash table.
        unmatched_left = left.rows * (1.0 - _match_fraction(left_ndv, right_ndv))
        unmatched_right = right.rows * (1.0 - _match_fraction(right_ndv, left_ndv))
        if node.join_type == "left":
            return max(inner + unmatched_left, left.rows), Provenance.DEFAULT
        if node.join_type == "right":
            return max(inner + unmatched_right, right.rows), Provenance.DEFAULT
        if node.join_type == "full":
            # |L ⟗ R| = |matched| + |unmatched L| + |unmatched R| >= max(|L|, |R|).
            total = inner + unmatched_left + unmatched_right
            return max(total, left.rows, right.rows), Provenance.DEFAULT
        return inner, Provenance.DEFAULT

    def _semi_anti_rows(
        self,
        node: Join,
        left: RelStats,
        right: RelStats,
        left_ndv: float | None,
        right_ndv: float | None,
    ) -> tuple[float, Provenance]:
        """Rows of a semi/anti join: the left rows whose key does (or does not) match.

        Under the containment assumption the fraction of `L`'s distinct keys present in
        `R` is ``min(1, d_R/d_L)``; under uniformity the same fraction of *rows* match.
        With either distinct count unmeasured the match fraction is unknowable, so both
        variants fall back to the upper bound `|L|` — over-budgeting memory rather than
        risking the under-estimate that would OOM the join's hash table.

        A **semi** join is additionally floored by skew: a hot left value that also appears
        in `R`'s measured MCV provably matches, so *all* `f_L(v)·|L|` of its rows survive —
        a firm lower bound the uniform ``d_R/d_L`` fraction can undercount on a skewed key.

        Semi and anti are derived from **one** semi estimate, because they partition `|L|`
        exactly: every left row either finds a match or does not. Estimating them
        independently (as this did) let both claim `|L|` whenever the distinct counts were
        unmeasured — a pair of estimates no execution can jointly produce — and priced a
        near-empty anti-join at full width. Deriving anti as `|L| - semi` makes the identity
        hold by construction, including the skew floor: rows *proved* to match by a shared hot
        value are exactly the rows proved not to survive the anti-join.
        """
        if not left_ndv or not right_ndv or left_ndv <= 0:
            # No usable distinct counts: either variant could keep everything, and `|L|` is the
            # tight upper bound for both. The complement identity is deliberately not applied
            # here — it would report one of the two as empty on no evidence.
            return left.rows, Provenance.DEFAULT
        matched = join_match_fraction(left_ndv, right_ndv)
        semi = max(0.0, min(left.rows, left.rows * matched))
        semi = min(left.rows, max(semi, self._semi_skew_floor(node, left, right)))
        if node.join_type == "semi":
            return semi, Provenance.DEFAULT
        return max(0.0, left.rows - semi), Provenance.DEFAULT

    def _semi_skew_floor(self, node: Join, left: RelStats, right: RelStats) -> float:
        """Left rows guaranteed to survive a semi-join because their (hot) key is in `R`.

        A value in *both* sides' measured MCV certainly exists in `R`, so every left row
        holding it matches — ``f_L(v)·|L|`` rows per shared hot value. Single-key only,
        capped at `|L|`."""
        if len(node.left_keys) != 1 or len(node.right_keys) != 1:
            return 0.0
        lstat = left.columns.get(node.left_keys[0])
        rstat = right.columns.get(node.right_keys[0])
        if lstat is None or rstat is None or not lstat.mcv or not rstat.mcv:
            return 0.0
        total = sum(f * left.rows for value, f in lstat.mcv.items() if value in rstat.mcv)
        return min(total, left.rows)

    def _inner_join_rows(
        self,
        node: Join,
        left: RelStats,
        right: RelStats,
        left_ndv: float | None,
        right_ndv: float | None,
    ) -> float:
        """Selinger containment: ``|L|x|R| / max(d_L, d_R)``, capped at the cartesian bound.

        PK-FK detection first: if one side's join key is (nearly) unique — its distinct
        count reaches its row count — then every row of the *other* side matches at most
        one row here, so under containment the result is ≈ the other side's rows. This is
        the dominant join shape (a fact table joined to a dimension on the dimension's
        primary key) and the Selinger ratio gets it badly wrong for a *composite* PK: the
        fact side's key-combination ndv is over-estimated (its columns are correlated),
        which deflates ``|L||R|/max(ndv)`` — TPC-H Q9's
        ``lineitem ⋈ partsupp ON (partkey, suppkey)`` was estimated 8x low, steering the
        join order into a needless multi-million-row intermediate. Single keys keep the
        ratio (their ndv is measured directly, so it is accurate); a non-saturated
        composite is a genuine many-to-many join the ratio models well.
        """
        skew = self._skew_matched_rows(node, left, right)
        if len(node.left_keys) >= 2 and _composite_pk_fk(
            left.rows, right.rows, left_ndv, right_ndv
        ):
            return max(left.rows, right.rows, skew)
        # With both sides' key frequencies measured, the join decomposes exactly into the
        # matched-hot-value term plus a uniform estimate over the *residual* mass. That sum is
        # sharper than either part alone: the uniform estimate alone prices a 47%-frequent key
        # as if it were average (a catastrophic under-estimate), while taking the skew term as
        # a mere floor discards the mass the MCV table does not cover.
        decomposed = self._mcv_join_rows(node, left, right, left_ndv, right_ndv)
        if decomposed is not None:
            return self._range_scaled(node, left, right, decomposed)
        ndvs = [v for v in (left_ndv, right_ndv) if v is not None and v > 0]
        if ndvs:
            # With only one side's ndv known, `max(d_L, d_R) >= d_known`, so dividing by
            # the known one over-estimates — the safe direction (over-budget, never OOM).
            selinger = min(left.rows * right.rows / max(ndvs), left.rows * right.rows)
            return self._range_scaled(node, left, right, max(selinger, skew))
        # No distinct counts at all: assume the key is ~unique on the smaller side, so the
        # result is ≈ the larger side.
        return max(left.rows, right.rows, skew)

    def _mcv_join_rows(
        self,
        node: Join,
        left: RelStats,
        right: RelStats,
        left_ndv: float | None,
        right_ndv: float | None,
    ) -> float | None:
        """The skew+residual decomposition of a single-key equi-join, or None.

        Single-key only: a composite key's *joint* frequency distribution is not measured, and
        multiplying per-column frequencies would assume an independence the columns of a real
        composite key never have.
        """
        if len(node.left_keys) != 1 or len(node.right_keys) != 1:
            return None
        lstat = left.columns.get(node.left_keys[0])
        rstat = right.columns.get(node.right_keys[0])
        if lstat is None or rstat is None:
            return None
        return mcv_join_rows(
            left.rows,
            right.rows,
            lstat.mcv,
            rstat.mcv,
            left_ndv or lstat.ndv,
            right_ndv or rstat.ndv,
        )

    def _range_scaled(self, node: Join, left: RelStats, right: RelStats, rows: float) -> float:
        """Scale a join estimate by how much the two key ranges actually overlap.

        Full disjointness is already special-cased as a provably-empty join. *Partial* overlap
        is the far more common shape — a three-year fact table joined to a one-year dimension,
        or either side narrowed by a date predicate — and every estimator above assumes the two
        key domains coincide. They do not: only the keys inside the intersection of the two
        `[min, max]` ranges can match, so under uniformity the estimate scales by the smaller
        side's overlapping fraction.

        Only trusted for ordinal bounds (`_ordinal`-mappable), for the same reason
        `_join_keys_range_disjoint` is: a string column's footer bounds may be byte-truncated,
        so an overlap computed from them is not sound. Never scales *up*, and never below the
        skew floor the measured hot values already prove.
        """
        factor = 1.0
        for lk, rk in zip(node.left_keys, node.right_keys, strict=False):
            lstat, rstat = left.columns.get(lk), right.columns.get(rk)
            if lstat is None or rstat is None:
                continue
            l_range = _ordinal_range(lstat)
            r_range = _ordinal_range(rstat)
            if l_range is None or r_range is None:
                continue
            lo, hi = max(l_range[0], r_range[0]), min(l_range[1], r_range[1])
            # The *narrower* side's overlap fraction is the binding one: whichever key domain
            # is smaller determines how much of the join can survive.
            fractions = [
                f
                for f in (
                    _overlap_share(lstat, l_range, lo, hi),
                    _overlap_share(rstat, r_range, lo, hi),
                )
                if f is not None
            ]
            if fractions:
                factor = min(factor, max(fractions))
        if factor >= 1.0:
            return rows
        floor = self._skew_matched_rows(node, left, right)
        return max(rows * factor, floor)

    def _skew_matched_rows(self, node: Join, left: RelStats, right: RelStats) -> float:
        """A lower bound on an equi-join's output from matching heavy-hitter key values.

        Selinger's uniform ``|L||R|/max(ndv)`` badly under-counts a **skewed** join key: a
        single value `v` present `f_L(v)` of the time on the left and `f_R(v)` on the right
        produces ``(f_L·|L|)·(f_R·|R|)`` output rows *by itself*, and a hot key (``cust_id = 7``
        at 47%) contributes ~``0.22·|L|·|R|`` that the uniform estimate misses entirely.
        Summing that cross product over the shared **measured** most-common values is a firm
        lower bound on the join size, so using it as a floor only *raises* the estimate — the
        safe direction, and precisely the one that keeps a skewed build side from OOMing.

        Single-key only (a composite key's joint frequency is not measured) and capped at the
        cartesian bound. Returns 0 when either side has no measured MCV for its key.
        """
        if len(node.left_keys) != 1 or len(node.right_keys) != 1:
            return 0.0
        lstat = left.columns.get(node.left_keys[0])
        rstat = right.columns.get(node.right_keys[0])
        if lstat is None or rstat is None or not lstat.mcv or not rstat.mcv:
            return 0.0
        total = 0.0
        for value, f_left in lstat.mcv.items():
            f_right = rstat.mcv.get(value)  # both keyed by str(value) of the same join key
            if f_right is not None:
                total += (f_left * left.rows) * (f_right * right.rows)
        return min(total, left.rows * right.rows)

    def _cartesian_rows(
        self, node: Join, left: RelStats, right: RelStats
    ) -> tuple[float, Provenance]:
        """Rows of a cross join, per join type — every left row matches every right row.

        `inner`/`left`/`right`/`full` all emit the full product `|L|x|R|` (with everything
        matched there are no unmatched rows for an outer join to add). A `semi` keeps each
        left row **once** when the right side is non-empty, and an `anti` keeps none — so they
        must not be given the product, which is why the cross-join shortcut is dispatched per
        type rather than applied blanketly.
        """
        if node.join_type == "semi":
            return (left.rows if right.rows > 0 else 0.0), Provenance.DEFAULT
        if node.join_type == "anti":
            return (0.0 if right.rows > 0 else left.rows), Provenance.DEFAULT
        return left.rows * right.rows, Provenance.DEFAULT

    def _is_cartesian(self, node: Join) -> bool:
        """Whether every key pair is a constant-on-both-sides pseudo-edge.

        A comma/cross join lowers to an equi-join on a synthetic `__cross_key` literal
        whose ndv is unmeasured, so the containment estimate fell through to
        ``max(|L|, |R|)`` — short of the true ``|L|x|R|`` by a factor of ``min(|L|, |R|)``.
        Applies to every join type (a `LEFT`/`FULL` cross join is just as large); the per-type
        row count is `_cartesian_rows`.
        """
        if not node.left_keys:
            return False
        return all(
            is_cartesian_key_pair(node.left, lk, node.right, rk)
            for lk, rk in zip(node.left_keys, node.right_keys, strict=False)
        )

    # --- shared metadata accessors ----------------------------------------
    def _stats_for(self, source_id: int) -> SourceStatistics | None:
        if self._source_stats is None or source_id >= len(self._source_stats):
            return None
        return self._source_stats[source_id]

    def expr_selectivity(self, predicate: Expr, over: RelStats | None = None) -> float:
        """Estimated fraction of rows `predicate` keeps, from the relation's column stats.

        The structural estimate only — unlike `_selectivity`, it is not attached to a
        `Filter` node and so cannot consult the per-signature measured ratio. Rules that
        reason about a *sub-expression* of a predicate (which has no plan signature of
        its own) use this.

        Args:
            predicate: A boolean scalar expression.
            over: The relation the predicate is applied to, whose column statistics
                supply the distinct counts, quantiles and skew values. Omitting it
                estimates from the predicate's structure and the cold-start constants
                alone — correct, just blunter.

        Returns:
            The estimated kept fraction, in `[0, 1]`.
        """
        stats = over if over is not None else RelStats(0.0, Provenance.DEFAULT)
        return predicate_selectivity(
            predicate,
            _ndvs(stats),
            self._cfg,
            _quantiles(stats),
            _mcvs(stats),
            _bounds(stats),
            _null_fractions(stats),
        )

    def _selectivity(self, node: Filter, child: RelStats) -> float:
        # A measured selectivity for this exact plan shape always wins (the
        # learning loop); otherwise estimate from the predicate's structure over the
        # *child's own* column statistics — the ones the scan seeded from this source.
        learned = self._learned.get(self._sig(node), {}).get("selectivity")
        if learned is not None:
            return learned
        return predicate_selectivity(
            node.predicate,
            _ndvs(child),
            self._cfg,
            _quantiles(child),
            _mcvs(child),
            _bounds(child),
            _null_fractions(child),
        )

    def _has_learned(self, node: LogicalPlan) -> bool:
        return "selectivity" in self._learned.get(self._sig(node), {})

    def _source_key(self, source_id: int) -> str | None:
        """The key bound source `source_id`'s learned statistics are filed under."""
        if source_id >= len(self._sources):
            return None
        return source_stats_key(self._sources[source_id])

    def learned_columns(self, source_id: int) -> dict[str, ColumnStat]:
        """The learned column statistics measured **for this source**, as `ColumnStat`s.

        This is the one place the four learned column maps (`__column_ndv__`,
        `__column_quantiles__`, `__column_mcv__`, `__column_avg_bytes__`) are read, and it
        reads them *sliced by the source's own identity*. Everything downstream — filter
        selectivity, join cardinality, group-by ndv, row width — then works from the
        statistics propagated on `RelStats.columns`, so a column's measured distribution
        travels with the relation it was measured from instead of being looked up by a
        name that two tables can share.
        """
        cached = self._learned_cols.get(source_id)
        if cached is not None:
            return cached
        key = self._source_key(source_id)
        ndv = columns_for(self._learned, NDV_KEY, key)
        quantiles = columns_for(self._learned, QUANTILES_KEY, key)
        mcv = columns_for(self._learned, MCV_KEY, key)
        widths = columns_for(self._learned, AVG_BYTES_KEY, key)
        cols: dict[str, ColumnStat] = {}
        for name in set(ndv) | set(quantiles) | set(mcv) | set(widths):
            measured = ndv.get(name)
            cols[name] = ColumnStat(
                ndv=float(measured) if measured and measured > 0 else None,
                quantiles=quantiles.get(name),
                mcv=mcv.get(name),
                avg_bytes=widths.get(name),
                # Measured by HLL/KLL/Misra-Gries — approximate by construction, so it may
                # inform cost and pruning but must never answer an exact `count_distinct`.
                provenance=Provenance.SKETCH,
            )
        self._learned_cols[source_id] = cols
        return cols

    def row_width(self, node: LogicalPlan, default: float) -> float:
        """Estimated average bytes per output row of `node`.

        A column's width is taken from the first source that knows it: a *learned*
        average byte width (measured, authoritative), else the width implied by its
        Arrow **type** (exact for fixed-width types, a documented prior for
        variable-length ones — see `plan.types.widths`), else the mean of this
        node's measured columns. Only a node with no schema at all falls back to
        `default`, the cost model's flat per-row constant.

        The schema floor matters because the byte axes gate broadcast eligibility.
        Costing every unmeasured relation at a flat per-row constant sized a
        two-`int64` join key (16 B/row) exactly like a 20-column payload, which
        over-sized narrow build sides by ~4x and forfeited their broadcast join.
        """
        widths = _avg_bytes(self.estimate(node))
        cols = node.available_columns()
        if not cols:
            return default
        schema = node.available_schema()
        typed: dict[str, float] = {}
        if schema is not None:
            typed = {f.name: column_bytes(f.type) for f in schema.arrow}
        measured = [widths[c] for c in cols if c in widths]
        if not measured and not typed:
            return self._opaque_width(node, default)
        # Neutral per-column filler for a column neither measured nor typed.
        known = measured or list(typed.values())
        avg_known = sum(known) / len(known)
        derived = sum(widths.get(c) or typed.get(c, avg_known) for c in cols)
        return max(derived, self._measured_scan_width(node))

    def _measured_scan_width(self, node: LogicalPlan) -> float:
        """Bytes per row the *source itself* measured, or `0.0` when it reported none.

        A connector that can answer `byte_size` and `row_count` has already measured the one
        quantity every type prior is trying to approximate, and nothing read it. That is the
        whole gap on unstructured and multimodal data: `io/formats/multimodal/media.py`
        reports the exact total size and file count from its listing — a directory of 200 MB
        videos is 200 MB per row, measured — while `column_bytes` could only offer the 36 B
        prior for the `binary` column it lands in. Five to six orders of magnitude, for a
        number already sitting in `SourceStatistics`.

        Read **only** from a connector that set `content_byte_size`, meaning its `byte_size`
        measures the rows' own content rather than their stored encoding. That flag is the
        whole safety argument, and it was earned by measurement rather than assumed: taking
        a *columnar* footer's `total_byte_size` as a floor as well moved TPC-H sf1's
        type-derived width from 88 to 142 B/row — closer to the true 139 — and made the
        benchmark **worse**, pushing dimension build sides past the broadcast threshold and
        taking q9 from 55.8 ms to 127.9 with ten other queries slower. A sharper estimate
        against a threshold tuned for the blunter one is a re-tuning, not an improvement.
        See `SourceStatistics.content_byte_size`.

        Taken as a floor (`max` with the type-derived sum) rather than as the answer, since a
        media listing's bytes are the *encoded* file and a decoded frame in memory is larger
        still — conservative in the right direction.

        Only at a `Scan`. Above one the projected columns differ, and the per-column widths
        propagated on `RelStats` are the right mechanism — attributing a whole relation's
        bytes to a single narrow projected column would invert the estimate.

        Args:
            node: The node whose width is being estimated.

        Returns:
            Measured bytes per row, or `0.0` when the source reported none.
        """
        if not isinstance(node, Scan):
            return 0.0
        stats = self._stats_for(node.source_id)
        if stats is None:
            return 0.0
        size, rows = stats.byte_size, stats.row_count
        if not getattr(stats, "content_byte_size", False) or not size or not rows or rows <= 0:
            return 0.0
        return float(size) / float(rows)

    def _opaque_width(self, node: LogicalPlan, default: float) -> float:
        """Row width for a node whose own schema says nothing — the `map_batches` case.

        A `MapBatches` is executed in Python and never lowered, so it publishes no output
        schema and its width fell all the way back to the flat `bytes_per_row` constant of
        64. That is the operator at the centre of every inference pipeline, and the rows
        flowing through it are the widest in the engine: a decoded 224x224x3 image is 147 KiB,
        so the estimate was off by three orders of magnitude for the memory envelope, the
        morsel cap, and the GPU batch seed all at once.

        The input's width is not a guess here. `MapBatches.available_columns` already
        implements the operator's stated contract — "if `output_columns` is omitted, the
        input columns are assumed to pass through" — so taking the *input's* width is the
        same assumption the plan already makes about the same operator, priced instead of
        counted. A declared `output_columns` means the shape genuinely changed, and there the
        flat default stands rather than a claim about columns the UDF invented.

        Deliberately **not** implemented by giving `MapBatches` an `available_schema`. That
        method feeds type inference and expression validation, where asserting the input's
        *types* survive a UDF that may rewrite them would turn an estimate into a wrong
        answer. A width only feeds cost and memory, where being closer is strictly better.

        Args:
            node: The node whose width could not be derived from its own schema.
            default: The flat per-row constant to fall back to.

        Returns:
            The input's estimated width for a pass-through `map_batches`, else `default`.
        """
        if isinstance(node, MapBatches) and node.output_columns is None:
            return self.row_width(node.input, default)
        return default

    def _side_ndv(self, keys: tuple[str, ...], side: RelStats) -> float | None:
        """Distinct count of one join side's key *set*, capped at its row count.

        The per-key counts come from **that side's own** propagated column statistics, so
        a key named `id` is answered by the distinct count of the `id` this side actually
        carries. Returns `None` when any key lacks a measured distinct count. See
        `combine_ndv` for why the counts are combined with damping rather than multiplied.
        """
        if not keys:
            return None
        ndv = _ndvs(side)
        if not all(k in ndv and ndv[k] > 0 for k in keys):
            return None
        return combine_ndv((ndv[k] for k in keys), side.rows)

    def _join_keys_range_disjoint(self, node: Join, left: RelStats, right: RelStats) -> bool:
        """Whether some equi-key pair has provably non-overlapping `[min, max]` ranges.

        Bounds are valid (if loose) supersets of the actual values, so two disjoint key
        ranges share no value and cannot match. Only numeric/date/decimal bounds are trusted
        (`_ordinal`-mappable): a string column's footer bounds may be *truncated*, so they
        are not a sound superset and a disjointness claim over them could be wrong. A single
        disjoint key pair is enough — an equi-join requires every key to match.
        """
        for lk, rk in zip(node.left_keys, node.right_keys, strict=False):
            lstat = left.columns.get(lk)
            rstat = right.columns.get(rk)
            if lstat is None or rstat is None:
                continue
            lo_l, hi_l = _ordinal(lstat.min), _ordinal(lstat.max)
            lo_r, hi_r = _ordinal(rstat.min), _ordinal(rstat.max)
            if None in (lo_l, hi_l, lo_r, hi_r):
                continue
            if hi_l < lo_r or hi_r < lo_l:  # the two ranges do not overlap
                return True
        return False


def _match_fraction(side_ndv: float | None, other_ndv: float | None) -> float:
    """The fraction of `side`'s rows that find a match, or 1.0 when it cannot be estimated.

    Defaulting to "everything matches" is what keeps an unmeasured outer join at its previous
    estimate: the unmatched term vanishes and the result falls back to `max(inner, |side|)`.
    Guessing a match fraction from nothing would invent null-extended rows.
    """
    if not side_ndv or not other_ndv or side_ndv <= 0:
        return 1.0
    return join_match_fraction(side_ndv, other_ndv)


def _overlap_share(
    stat: ColumnStat, own: tuple[float, float], lo: float, hi: float
) -> float | None:
    """The share of one key column that lies inside `[lo, hi]`.

    Measured as **mass** when the column carries a quantile grid — `F(hi) - F(lo)`, the
    fraction of its rows inside the intersection — and only as a share of the `[min, max]`
    *width* when it does not.

    The distinction matters exactly where the estimate does. A fact table's date key spanning
    three years joined to a dimension covering the most recent one overlaps in a third of its
    *width*, but if the fact table's rows are concentrated in that recent year it overlaps in
    most of its *mass* — and the width-based factor would then cut the join estimate to a
    third of the truth. Real key distributions are rarely uniform over their range, which is
    the whole reason the learning loop measures a quantile grid in the first place.
    """
    if hi < lo:
        return 0.0
    grid = stat.quantiles
    if grid:
        below_hi = _fraction_below_quantiles(hi, grid)
        below_lo = _fraction_below_quantiles(lo, grid)
        if below_hi is not None and below_lo is not None:
            return max(0.0, min(1.0, below_hi - below_lo))
    return overlap_fraction(own, (lo, hi))


def _ordinal_range(stat: ColumnStat) -> tuple[float, float] | None:
    """A column's `[min, max]` as a pair of ordinals, or None when it is not comparable."""
    lo, hi = _ordinal(stat.min), _ordinal(stat.max)
    if lo is None or hi is None or hi < lo:
        return None
    return lo, hi


def _ndvs(stats: RelStats) -> dict[str, float]:
    """`{column: ndv}` for every column of `stats` whose distinct count is known.

    Read off the relation's *propagated* statistics rather than a global name-keyed map,
    which is what binds a distinct count to the column it was actually measured from.
    """
    return {
        name: float(col.ndv)
        for name, col in stats.columns.items()
        if col.ndv is not None and col.ndv > 0
    }


def _quantiles(stats: RelStats) -> dict[str, Any]:
    """`{column: {"probs": [...], "values": [...]}}` for every column with a quantile grid.

    Feeds `predicate_selectivity`'s histogram interpolation for range predicates.
    """
    return {name: col.quantiles for name, col in stats.columns.items() if col.quantiles}


def _mcvs(stats: RelStats) -> dict[str, dict[str, float]]:
    """`{column: {str(value): frequency}}` for every column with measured top values.

    Feeds `predicate_selectivity`'s skew-aware equality estimate, which is far sharper
    than a uniform `1/ndv` on exactly the columns where uniformity is most wrong.
    """
    return {name: dict(col.mcv) for name, col in stats.columns.items() if col.mcv}


def _avg_bytes(stats: RelStats) -> dict[str, float]:
    """`{column: bytes/row}` for every column with a measured average width.

    Makes the cost model's memory/IO/broadcast axes byte-true for wide columns (large
    strings, embeddings, blob handles), where a flat per-row constant is off by orders.
    """
    return {
        name: float(col.avg_bytes)
        for name, col in stats.columns.items()
        if col.avg_bytes is not None and col.avg_bytes > 0
    }


def _null_fractions(stats: RelStats) -> dict[str, float]:
    """`{column: null_count / rows}` for every column whose null count is known.

    SQL keeps only rows where a predicate is TRUE, so a column's null mass is dropped by
    both `p` and `NOT p`. `predicate_selectivity` subtracts it from the complement, and a
    *measured* null count (a Parquet footer carries one for every column) makes that
    subtraction exact instead of the `null_selectivity` prior.
    """
    if stats.rows <= 0:
        return {}
    return {
        name: min(1.0, max(0.0, col.null_count / stats.rows))
        for name, col in stats.columns.items()
        if col.null_count is not None
    }


def _bounds(stats: RelStats) -> dict[str, tuple[Any, Any]]:
    """`{column: (min, max)}` for every column of `stats` that knows both bounds.

    Feeds `predicate_selectivity`'s uniformity fallback for range predicates. Bounds
    survive filters and joins as valid (if loose) bounds, so this is available on any
    relation whose source declared column statistics — from the very first query, before
    the learning loop has measured anything.
    """
    return {
        name: (col.min, col.max)
        for name, col in stats.columns.items()
        if col.min is not None and col.max is not None
    }


def combine_ndv(per_column: Iterable[float], cap: float) -> float:
    """Distinct count of a *set* of columns, from each column's distinct count.

    The independence assumption gives the product `∏ d_i`, but real key sets are
    correlated — `(city, state)` has ~`d_city` combinations, not `d_city x d_state`; a
    composite primary key's columns are correlated by construction. The product therefore
    overshoots, and since it is capped at the relation's row count it simply *saturates*,
    telling the optimizer that grouping (or joining on the key set) reduces nothing.

    The counts are instead combined with **exponential backoff**: the largest at full
    weight, each subsequent one damped by a further square root. The result lies between
    `max_i d_i` (the perfectly-correlated / functional-dependence floor, where the extra
    columns add nothing) and `∏ d_i` (the independence ceiling), which are exactly the
    Fréchet bounds for the combination. A single column is returned unchanged.

    The cap is the relation's row count: a learned ndv reflects the *unfiltered* source,
    but a relation cannot hold more distinct key combinations than it has rows.

    This is the one definition used for join keys, group-by keys, and `DISTINCT` column
    sets — they are the same quantity, and estimating them differently made a group-by
    saturate to its input size while the join on the same columns did not.

    One case is not a heuristic at all and is handled exactly: if any single column is
    already a **key** of the relation — its distinct count reaches the row count — then it
    functionally determines every other column, so the set has exactly that many distinct
    combinations. Backoff would instead push the estimate above the row count and let the cap
    clamp it, which looks identical for the count but hides the certainty; more importantly it
    is the shape a surrogate-key group-by (`GROUP BY order_id, order_date`) always takes, and
    the exact answer is what makes the group-by estimate stop depending on the number of
    incidental columns dragged along with the key.

    Args:
        per_column: Each column's distinct count. Non-positive counts are ignored.
        cap: The relation's row count.

    Returns:
        The estimated number of distinct combinations, in `[1, cap]`.
    """
    ordered = sorted((d for d in per_column if d > 0), reverse=True)
    if not ordered:
        return 1.0
    if cap > 0 and ordered[0] >= _UNIQUE_KEY_NDV_RATIO * cap:
        return max(1.0, min(ordered[0], cap))  # a key determines the rest
    combined = 1.0
    exponent = 1.0
    for d in ordered:
        combined *= d**exponent
        exponent /= 2.0
    return max(1.0, min(combined, cap))


# A composite join key is treated as unique (a candidate key) when its distinct-count
# estimate reaches this fraction of its rows — allowing for the ndv sketch's ~1% error
# and a lightly-filtered dimension.
_UNIQUE_KEY_NDV_RATIO = 0.95


def _composite_pk_fk(
    left_rows: float,
    right_rows: float,
    left_ndv: float | None,
    right_ndv: float | None,
) -> bool:
    """Whether a composite-key join is many-to-one (one side's key plausibly unique).

    True when either side's (capped) combination ndv saturates its row count — that
    side's composite key is then ~unique, so each row of the other side matches at most
    one, and the result is the FK side's rows (the caller uses `max(left, right)`).
    """

    def saturated(ndv: float | None, rows: float) -> bool:
        return ndv is not None and rows > 0 and ndv >= _UNIQUE_KEY_NDV_RATIO * rows

    return saturated(left_ndv, left_rows) or saturated(right_ndv, right_rows)


def _join_provably_empty(join_type: str, left: RelStats, right: RelStats) -> bool:
    """Whether an equi-join's result is provably empty from EXACT-empty input(s).

    Per join type, the result has zero rows when:

    - ``inner`` / ``semi`` — *either* side is empty (no row can match);
    - ``left`` / ``anti``  — the *left* side is empty (the output is left-driven);
    - ``right``            — the *right* side is empty;
    - ``full``             — *both* sides are empty (each side's rows are preserved).

    Only an EXACT-empty input proves emptiness; a merely-estimated zero does not.
    """
    left_empty = left.rows_exact and left.rows == 0
    right_empty = right.rows_exact and right.rows == 0
    if join_type in ("inner", "semi"):
        return left_empty or right_empty
    if join_type in ("left", "anti"):
        return left_empty
    if join_type == "right":
        return right_empty
    if join_type == "full":
        return left_empty and right_empty
    return False


def _canonical_sort_prefix(keys: tuple) -> tuple[str, ...]:
    """The leading run of sort keys that establish a *canonical* ordering.

    `RelStats.sorted_by` records ascending, nulls-last column orderings only — the
    one ordering a `Sort` (or a source declaring sortedness) and a consumer can
    compare unambiguously. A key that is a non-column expression, descending, or
    nulls-first stops the prefix: the ordering past it is not a plain column prefix
    we can soundly claim. (A connector that sets `SourceStatistics.sorted_by`
    asserts this same ascending/nulls-last contract.)
    """
    out: list[str] = []
    for k in keys:
        if not isinstance(k.expr, Col) or k.descending or k.nulls_first:
            break
        out.append(k.expr.name)
    return tuple(out)


def _signature(node: LogicalPlan) -> str:
    """A structural signature of a node (ignoring literal values), for learning."""
    from batcher.kyber.signature import plan_signature

    return plan_signature(node)
