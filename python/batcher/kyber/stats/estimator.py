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

from batcher.config import CardinalityConfig, active_config
from batcher.kyber.learning import (
    AVG_BYTES_KEY,
    CARDINALITY_CORRECTION_KEY,
    MCV_KEY,
    NDV_KEY,
    QUANTILES_KEY,
)
from batcher.kyber.stats import columns as col_prop
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.plan.expr_ir import Col, Expr, Lit
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
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    Window,
    is_cartesian_key_pair,
)
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import Provenance, RelStats, weakest

__all__ = ["StatsEstimator", "combine_ndv"]

# Operators whose estimates carry a learned correction. Joins, aggregates, and distincts
# are where structural cardinality estimation is worst (containment assumptions, ndv
# products) and where being wrong is most expensive — a mis-sized build side or shuffle.
# `Unnest` belongs for a different reason: its fan-out is the average list length, which
# is a property of the *data*, not the plan — no structural rule can know it, so the
# estimator passes the child count through and is wrong by exactly that factor (a RAG
# pipeline chunking documents 20-to-1 under-sizes every stage below it). Measuring it
# once and correcting is the only way, and is precisely the Core-measures/Kyber-consumes
# loop.
# `Filter` is deliberately excluded: its measured *selectivity* is already learned
# per-signature, and correcting it here as well would count the same error twice.
# Row-preserving operators (Project, Sort, Limit) need no correction — they inherit it
# from their input.
_CORRECTABLE = (Aggregate, Distinct, Join, Unnest)


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
        # per node in `_annotate_ops`, plus every cost-based rule), so planning is
        # super-linear in plan depth and `plan_signature` re-hashes whole subtrees.
        # Each entry holds a strong reference to its keyed node alongside the value so
        # a freed node's reused `id()` can never produce a stale hit.
        self._row_cache: dict[int, tuple[LogicalPlan, RelStats]] = {}
        self._sig_cache: dict[int, tuple[LogicalPlan, str]] = {}
        # Merged column→ndv map (source-stats NDV seeded under learned NDV), built once
        # on first access since `_ndv` is read on every selectivity / join estimate.
        self._ndv_cache: dict[str, float] | None = None

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
        # Learned-first: trust a measured absolute size for this exact shape — except
        # a Filter, whose measured *selectivity* ratio (applied below to the current
        # input) generalizes across input sizes better than a stale absolute count.
        if not self._exact_first:
            learned = self._learned.get(self._sig(node))
            if learned is not None and "rows" in learned and not isinstance(node, Filter):
                return RelStats(float(learned["rows"]), Provenance.LEARNED)

        if isinstance(node, Scan):
            return self._estimate_scan(node)
        if isinstance(node, Filter):
            return self._estimate_filter(node)
        if isinstance(node, Project):
            child = self.estimate(node.input)
            columns = col_prop.project_columns(node.items, child, node.input.available_schema())
            return RelStats(child.rows, child.provenance, columns)
        if isinstance(node, MapBatches):
            # Row-preserving (map_batches may change rows, but assume 1:1); the
            # opaque UDF means output columns are unknown.
            return RelStats(self.estimate(node.input).rows, Provenance.DEFAULT)
        if isinstance(node, Unnest):
            # Explode multiplies rows by the average list length — a property of the data
            # that no structural rule can know, so the neutral (1x) default here is wrong
            # by exactly that factor on the first run. `Unnest` is `_CORRECTABLE`, so the
            # measured fan-out from a previous run is applied by `estimate` on top of this.
            return RelStats(self.estimate(node.input).rows, Provenance.DEFAULT)
        if isinstance(node, Unpivot):
            # Unpivot emits one row per `on` column — an exact, data-independent fan-out.
            child = self.estimate(node.input)
            rows = child.rows * max(1, len(node.on))
            return RelStats(rows, child.provenance)
        if isinstance(node, Sample):
            child = self.estimate(node.input)
            # Fixed-count sample yields exactly min(n, input); fraction scales the input.
            rows = (
                min(child.rows, float(node.n)) if node.n is not None else child.rows * node.fraction
            )
            return RelStats(rows, Provenance.DEFAULT, col_prop.sample_columns(child))
        if isinstance(node, Aggregate):
            return self._estimate_aggregate(node)
        if isinstance(node, Sort):
            return self._estimate_sort(node)
        if isinstance(node, Window):
            # Row-preserving: Window appends columns, never changes the row count, so
            # the input columns' stats (EXACT included) carry through untouched.
            child = self.estimate(node.input)
            return RelStats(
                child.rows, child.provenance, col_prop.window_columns(child), child.sorted_by
            )
        if isinstance(node, Limit):
            return self._estimate_limit(node)
        if isinstance(node, Distinct):
            return self._estimate_distinct(node)
        if isinstance(node, Union):
            return self._estimate_union(node)
        if isinstance(node, Join):
            return self._estimate_join(node)
        if isinstance(node, AsofJoin):
            # ASOF is left-style: exactly one output row per left row, so the count
            # (and its provenance) is the left input's — EXACT when the left is, so
            # `asof_join(...).count()` answers from metadata (incl. an empty left → 0).
            left = self.estimate(node.left)
            return RelStats(left.rows, left.provenance)
        return RelStats(self._cfg.unknown_rows, Provenance.DEFAULT)

    # --- per-operator estimators ------------------------------------------
    def _estimate_scan(self, node: Scan) -> RelStats:
        src_stats = self._stats_for(node.source_id)
        if src_stats is not None:
            base = src_stats.to_relstats(default_rows=self._cfg.unknown_rows)
            columns = col_prop.scan_columns(base.columns, self._ndv)
            return RelStats(base.rows, base.provenance, columns, base.sorted_by)
        # Sources may be absent (plan-shape optimization with no bound inputs) or
        # duck-typed without `row_count`; treat either as unknown rather than crash.
        source = self._sources[node.source_id] if node.source_id < len(self._sources) else None
        row_count_fn = getattr(source, "row_count", None)
        n = row_count_fn() if callable(row_count_fn) else None
        columns = col_prop.scan_columns({}, self._ndv)
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
        sel = self._selectivity(node, child)
        # `prov` is LEARNED (measured selectivity) or DEFAULT (Selinger) — never
        # EXACT — so a filtered row count is never EXACT, however exact the child.
        prov = Provenance.LEARNED if self._has_learned(node) else Provenance.DEFAULT
        return RelStats(
            child.rows * sel,
            weakest(child.provenance, prov),
            col_prop.filter_columns(child),
            child.sorted_by,
        )

    def _estimate_sort(self, node: Sort) -> RelStats:
        child = self.estimate(node.input)
        rows = child.rows
        prov = child.provenance
        if node.limit is not None:
            rows = min(rows, float(node.limit))
        # Sort preserves the exact value set, so column stats pass through unchanged.
        return RelStats(rows, prov, dict(child.columns), _canonical_sort_prefix(node.keys))

    def _estimate_limit(self, node: Limit) -> RelStats:
        child = self.estimate(node.input)
        rows = min(child.rows, float(node.n))
        # `Limit(x, 0)` is provably empty regardless of the child's provenance (it is
        # the canonical empty marker), so its zero row count is EXACT — letting
        # `count()`/`is_empty()` answer a pruned-to-empty subtree from metadata even
        # over an unknown source. Otherwise the (possibly truncated) count is as exact
        # as the child.
        prov = Provenance.EXACT if node.n == 0 else child.provenance
        return RelStats(rows, prov, col_prop.limit_columns(child), child.sorted_by)

    def _estimate_union(self, node: Union) -> RelStats:
        children = [self.estimate(i) for i in node.inputs]
        total = sum(c.rows for c in children)
        prov = weakest(*(c.provenance for c in children)) if children else Provenance.DEFAULT
        names = node.available_columns()
        columns = col_prop.union_columns(children, names)
        if node.distinct:
            # Dedup across branches: row count is no longer exact (overlap unknown).
            return RelStats(total, weakest(prov, Provenance.DEFAULT), columns)
        return RelStats(total, prov, columns)

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
        ndv = self._ndv
        key_ndvs: list[float] = []
        for key in node.group_keys:
            if isinstance(key.expr, Col) and key.expr.name in ndv and ndv[key.expr.name] > 0:
                key_ndvs.append(ndv[key.expr.name])
            else:
                # An unknown-placeholder input (an uncountable source — `from_batches`,
                # a stream, an un-pushed SQL scan) must NOT be shrunk below the
                # "unknown" threshold: the shrunk guess (0.1·unknown) is small enough
                # to look like a real estimate, so the optimizer would *budget* it and
                # Carbonite could wrongly reject an actually-small query as infeasible.
                # Keep it a placeholder so it stays unbudgeted (a guess never fails a
                # real query — the documented admission contract).
                if child.rows >= self._cfg.unknown_rows:
                    return RelStats(child.rows, Provenance.DEFAULT, key_cols)
                return RelStats(max(1.0, child.rows * 0.1), Provenance.DEFAULT, key_cols)
        # The distinct combinations of the group-key set — the same quantity a join
        # computes for its key set, so the same (damped) combiner. Multiplying the
        # per-key counts assumed independence; correlated keys then saturated the cap and
        # the optimizer concluded that grouping reduced nothing.
        groups = combine_ndv(key_ndvs, child.rows)
        return RelStats(groups, Provenance.LEARNED, key_cols)

    def _estimate_distinct(self, node: Distinct) -> RelStats:
        """Dedup count ≈ the distinct combinations of the projected columns.

        The same quantity `Aggregate` estimates for its group keys, so the same
        `combine_ndv` combiner. For the common single-column `DISTINCT col` this is the
        column's measured ndv (~exact); a multi-column set is damped rather than
        multiplied, since the columns of a real key set are correlated. Falls back to 50%
        when any column's ndv is unmeasured."""
        child = self.estimate(node.input)
        cols = node.available_columns()
        ndv = self._ndv
        columns = col_prop.distinct_columns(child)
        if cols and all(c in ndv and ndv[c] > 0 for c in cols):
            groups = combine_ndv((ndv[c] for c in cols), child.rows)
            return RelStats(groups, Provenance.LEARNED, columns)
        # Unknown-placeholder input → keep the placeholder (see `_estimate_aggregate`):
        # shrinking it would let admission wrongly reject a small query.
        if child.rows >= self._cfg.unknown_rows:
            return RelStats(child.rows, Provenance.DEFAULT, columns)
        return RelStats(max(1.0, child.rows * 0.5), Provenance.DEFAULT, columns)

    def _estimate_join(self, node: Join) -> RelStats:
        left = self.estimate(node.left)
        right = self.estimate(node.right)
        # Provable emptiness wins over any learned/Selinger estimate: when the
        # relevant side(s) are EXACT-empty the join is EXACT-empty too, so
        # `count()`/`is_empty()` answer 0 from metadata without executing the join.
        if _join_provably_empty(node.join_type, left, right):
            return RelStats(0.0, Provenance.EXACT)
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
            return left.rows * right.rows, Provenance.DEFAULT

        left_ndv = self._side_ndv(node.left_keys, left.rows)
        right_ndv = self._side_ndv(node.right_keys, right.rows)

        if node.join_type in ("semi", "anti"):
            return self._semi_anti_rows(node.join_type, left, left_ndv, right_ndv)

        inner = self._inner_join_rows(node, left, right, left_ndv, right_ndv)
        if node.join_type == "left":
            return max(inner, left.rows), Provenance.DEFAULT
        if node.join_type == "right":
            return max(inner, right.rows), Provenance.DEFAULT
        if node.join_type == "full":
            # |L ⟗ R| = |matched| + |unmatched L| + |unmatched R| >= max(|L|, |R|).
            return max(inner, left.rows, right.rows), Provenance.DEFAULT
        return inner, Provenance.DEFAULT

    def _semi_anti_rows(
        self, join_type: str, left: RelStats, left_ndv: float | None, right_ndv: float | None
    ) -> tuple[float, Provenance]:
        """Rows of a semi/anti join: the left rows whose key does (or does not) match.

        Under the containment assumption the fraction of `L`'s distinct keys present in
        `R` is ``min(1, d_R/d_L)``; under uniformity the same fraction of *rows* match.
        With either distinct count unmeasured the match fraction is unknowable, so both
        variants fall back to the upper bound `|L|` — over-budgeting memory rather than
        risking the under-estimate that would OOM the join's hash table.
        """
        if not left_ndv or not right_ndv or left_ndv <= 0:
            return left.rows, Provenance.DEFAULT
        matched = min(1.0, right_ndv / left_ndv)
        fraction = matched if join_type == "semi" else 1.0 - matched
        return max(0.0, left.rows * fraction), Provenance.DEFAULT

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
        if len(node.left_keys) >= 2 and _composite_pk_fk(
            left.rows, right.rows, left_ndv, right_ndv
        ):
            return max(left.rows, right.rows)
        ndvs = [v for v in (left_ndv, right_ndv) if v is not None and v > 0]
        if ndvs:
            # With only one side's ndv known, `max(d_L, d_R) >= d_known`, so dividing by
            # the known one over-estimates — the safe direction (over-budget, never OOM).
            return min(left.rows * right.rows / max(ndvs), left.rows * right.rows)
        # No distinct counts at all: assume the key is ~unique on the smaller side, so the
        # result is ≈ the larger side.
        return max(left.rows, right.rows)

    def _is_cartesian(self, node: Join) -> bool:
        """Whether every key pair is a constant-on-both-sides pseudo-edge.

        A comma/cross join lowers to an equi-join on a synthetic `__cross_key` literal
        whose ndv is unmeasured, so the containment estimate fell through to
        ``max(|L|, |R|)`` — short of the true ``|L|x|R|`` by a factor of ``min(|L|, |R|)``.
        """
        if not node.left_keys or node.join_type != "inner":
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

    def expr_selectivity(self, predicate: Expr) -> float:
        """Estimated fraction of rows `predicate` keeps, from the learned column stats.

        The structural estimate only — unlike `_selectivity`, it is not attached to a
        `Filter` node and so cannot consult the per-signature measured ratio. Rules that
        reason about a *sub-expression* of a predicate (which has no plan signature of
        its own) use this.

        Args:
            predicate: A boolean scalar expression.

        Returns:
            The estimated kept fraction, in `[0, 1]`.
        """
        return predicate_selectivity(predicate, self._ndv, self._cfg, self._quantiles, self._mcv)

    def _selectivity(self, node: Filter, child: RelStats) -> float:
        # A measured selectivity for this exact plan shape always wins (the
        # learning loop); otherwise estimate from the predicate's structure.
        learned = self._learned.get(self._sig(node), {}).get("selectivity")
        if learned is not None:
            return learned
        return predicate_selectivity(
            node.predicate,
            self._ndv,
            self._cfg,
            self._quantiles,
            self._mcv,
            _bounds(child),
            _null_fractions(child),
        )

    def _has_learned(self, node: LogicalPlan) -> bool:
        return "selectivity" in self._learned.get(self._sig(node), {})

    @property
    def _ndv(self) -> dict[str, float]:
        """Per-column distinct counts (column name → ndv), used to sharpen equality
        selectivity to `1/ndv` and to estimate join cardinality as `|L||R|/max(ndv)`.

        Seeded from **source statistics** (footer / written-file HLL sketches carried by
        `SourceStatistics.columns`) and overlaid with **learned** ndv from measured runs
        (which wins, being workload-true). Source NDV is what gives a *cold* join — before
        any run has been measured — an NDV-based cardinality instead of the
        `max(left, right)` fallback that mis-estimates a low-NDV many-to-many join by
        orders of magnitude and steers join order into huge intermediates.
        """
        if self._ndv_cache is None:
            merged: dict[str, float] = {}
            for st in self._source_stats or ():
                if st is None:
                    continue
                for name, col in st.columns.items():
                    if col.ndv is not None and col.ndv > 0:
                        merged[name] = float(col.ndv)
            merged.update(self._learned.get(NDV_KEY, {}))  # measured wins
            self._ndv_cache = merged
        return self._ndv_cache

    @property
    def _quantiles(self) -> dict[str, Any]:
        """Learned per-column quantile boundaries
        (`{col: {"probs": [...], "values": [...]}}`, both ascending), used for
        histogram-based range selectivity. Empty until the metadata loop fills it."""
        return self._learned.get(QUANTILES_KEY, {})

    @property
    def _mcv(self) -> dict[str, dict[str, float]]:
        """Learned per-column most-common-values (`{col: {str(value): frequency}}`),
        used to sharpen equality selectivity on skewed columns to the value's measured
        frequency. Empty until the metadata loop fills it."""
        return self._learned.get(MCV_KEY, {})

    @property
    def _avg_bytes(self) -> dict[str, float]:
        """Learned per-column average byte widths (column name → bytes/row),
        measured from `ColumnStats.avg_byte_width`. Empty until the metadata loop
        fills it; this is what turns the cost model's memory/IO/broadcast axes
        byte-true for wide columns (large strings, embeddings, blob handles)."""
        return self._learned.get(AVG_BYTES_KEY, {})

    def row_width(self, node: LogicalPlan, default: float) -> float:
        """Estimated average bytes per output row of `node`.

        Sums the node's output columns' *learned* average byte widths; columns
        with no measured width contribute the mean of the measured ones (a neutral
        per-column estimate). When **no** output column has a learned width yet,
        falls back to `default` — the cost model's flat per-row constant — so a
        cold-start plan costs exactly as it did before byte-awareness.
        """
        widths = self._avg_bytes
        cols = node.available_columns()
        measured = [widths[c] for c in cols if c in widths]
        if not measured:
            return default
        avg_known = sum(measured) / len(measured)
        return sum(widths.get(c, avg_known) for c in cols)

    def _side_ndv(self, keys: tuple[str, ...], rows: float) -> float | None:
        """Distinct count of one join side's key *set*, capped at its row count.

        Returns `None` when any key lacks a learned distinct count. See `combine_ndv` for
        why the per-key counts are combined with damping rather than multiplied.
        """
        if not keys:
            return None
        ndv = self._ndv
        if not all(k in ndv and ndv[k] > 0 for k in keys):
            return None
        return combine_ndv((ndv[k] for k in keys), rows)

    def input_sizes(self, node: Join) -> tuple[RelStats, RelStats]:
        """The estimated sizes of a join's two inputs (for build-side choice)."""
        return self.estimate(node.left), self.estimate(node.right)


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

    Args:
        per_column: Each column's distinct count. Non-positive counts are ignored.
        cap: The relation's row count.

    Returns:
        The estimated number of distinct combinations, in `[1, cap]`.
    """
    ordered = sorted((d for d in per_column if d > 0), reverse=True)
    if not ordered:
        return 1.0
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
