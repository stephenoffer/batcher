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

from typing import Any

from batcher.config import CardinalityConfig, active_config
from batcher.kyber.stats import columns as col_prop
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.plan.expr_ir import Col
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
)
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import Provenance, RelStats, weakest

__all__ = ["StatsEstimator"]


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
        result = self._estimate_uncached(node)
        self._row_cache[id(node)] = (node, result)
        return result

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
            return RelStats(
                child.rows, child.provenance, col_prop.project_columns(node.items, child)
            )
        if isinstance(node, MapBatches):
            # Row-preserving (map_batches may change rows, but assume 1:1); the
            # opaque UDF means output columns are unknown.
            return RelStats(self.estimate(node.input).rows, Provenance.DEFAULT)
        if isinstance(node, Unnest):
            # Explode multiplies rows by the (data-dependent) average list length.
            # Without a learned fan-out we keep the child estimate as a neutral default.
            return RelStats(self.estimate(node.input).rows, Provenance.DEFAULT)
        if isinstance(node, Unpivot):
            # Unpivot emits one row per `on` column — an exact, data-independent fan-out.
            child = self.estimate(node.input)
            rows = child.rows * max(1, len(node.on))
            return RelStats(rows, child.provenance)
        if isinstance(node, Sample):
            child = self.estimate(node.input).rows
            # Fixed-count sample yields exactly min(n, input); fraction scales the input.
            rows = min(child, float(node.n)) if node.n is not None else child * node.fraction
            return RelStats(rows, Provenance.DEFAULT)
        if isinstance(node, Aggregate):
            return self._estimate_aggregate(node)
        if isinstance(node, Sort):
            return self._estimate_sort(node)
        if isinstance(node, Window):
            # Row-preserving: Window appends columns, never changes the row count.
            child = self.estimate(node.input)
            return RelStats(child.rows, child.provenance, dict(child.columns), child.sorted_by)
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
        sel = self._selectivity(node)
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
        if not self._exact_first:
            learned_rows = self._learned.get(self._sig(node), {}).get("rows")
            if learned_rows is not None:
                return RelStats(float(learned_rows), Provenance.LEARNED)
        ndv = self._ndv
        groups = 1.0
        for key in node.group_keys:
            if isinstance(key.expr, Col) and key.expr.name in ndv and ndv[key.expr.name] > 0:
                groups *= ndv[key.expr.name]
            else:
                # An unknown-placeholder input (an uncountable source — `from_batches`,
                # a stream, an un-pushed SQL scan) must NOT be shrunk below the
                # "unknown" threshold: the shrunk guess (0.1·unknown) is small enough
                # to look like a real estimate, so the optimizer would *budget* it and
                # Carbonite could wrongly reject an actually-small query as infeasible.
                # Keep it a placeholder so it stays unbudgeted (a guess never fails a
                # real query — the documented admission contract).
                if child.rows >= self._cfg.unknown_rows:
                    return RelStats(child.rows, Provenance.DEFAULT)
                return RelStats(max(1.0, child.rows * 0.1), Provenance.DEFAULT)
        return RelStats(max(1.0, min(groups, child.rows)), Provenance.LEARNED)

    def _estimate_distinct(self, node: Distinct) -> RelStats:
        """Dedup count ≈ distinct value combinations — the product of the columns'
        learned ndv (capped at input), the same metadata `Aggregate` uses. For the
        common single-column `DISTINCT col` this is ~exact; multi-column is a capped
        upper bound. Falls back to 50% when any column's ndv is unmeasured."""
        child = self.estimate(node.input)
        cols = node.available_columns()
        ndv = self._ndv
        columns = col_prop.distinct_columns(child)
        if cols and all(c in ndv and ndv[c] > 0 for c in cols):
            groups = 1.0
            for c in cols:
                groups *= ndv[c]
            return RelStats(max(1.0, min(groups, child.rows)), Provenance.LEARNED, columns)
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
        if not self._exact_first:
            learned_rows = self._learned.get(self._sig(node), {}).get("rows")
            if learned_rows is not None:
                return RelStats(float(learned_rows), Provenance.LEARNED)
        if node.join_type in {"semi", "anti"}:
            return RelStats(left.rows, Provenance.DEFAULT)
        # PK–FK detection first: if one side's join key is (nearly) unique — its
        # distinct count reaches its row count — then every row of the *other* side
        # matches at most one row here, so under containment the result is ≈ the other
        # side's rows. This is the dominant join shape (a fact table joined to a
        # dimension on the dimension's primary key) and the Selinger ratio below gets
        # it badly wrong for a *composite* PK: the fact side's key-combination ndv is
        # over-estimated (its columns are correlated), which deflates `|L||R|/max(ndv)`
        # — TPC-H Q9's `lineitem ⋈ partsupp ON (partkey, suppkey)` was estimated 8× low,
        # steering the join order into a needless multi-million-row intermediate.
        left_ndv = self._side_ndv(node.left_keys, left.rows)
        right_ndv = self._side_ndv(node.right_keys, right.rows)
        # Many-to-one (PK–FK) detection for *composite* keys. A composite key's
        # combination ndv is the damped product capped at the side's rows; when it
        # saturates (≈ rows) that side's key is plausibly unique, so the join is
        # many-to-one and the FK side is preserved — the result is ≈ the larger input.
        # The Selinger ratio below would instead *under*-estimate this badly (the fact
        # side's correlated key columns inflate its combination ndv, deflating the
        # divisor's effect), which steered TPC-H Q9 into a needless 6M-row intermediate.
        # Single keys keep the ratio (their ndv is measured directly, so it is accurate);
        # a non-saturated composite is a genuine many-to-many join the ratio models well.
        if len(node.left_keys) >= 2 and _composite_pk_fk(
            left.rows, right.rows, left_ndv, right_ndv
        ):
            return RelStats(max(left.rows, right.rows), Provenance.DEFAULT)
        # Classic equi-join estimate: |L⋈R| ≈ |L|·|R| / max(ndv_lk, ndv_rk) when a
        # single key's distinct count is known. Without it, assume the key is
        # ~unique on the smaller side, so the result ≈ the larger side.
        ndvs = [v for v in (left_ndv, right_ndv) if v is not None and v > 0]
        if ndvs:
            return RelStats(left.rows * right.rows / max(ndvs), Provenance.DEFAULT)
        return RelStats(max(left.rows, right.rows), Provenance.DEFAULT)

    # --- shared metadata accessors ----------------------------------------
    def _stats_for(self, source_id: int) -> SourceStatistics | None:
        if self._source_stats is None or source_id >= len(self._source_stats):
            return None
        return self._source_stats[source_id]

    def _selectivity(self, node: Filter) -> float:
        # A measured selectivity for this exact plan shape always wins (the
        # learning loop); otherwise estimate from the predicate's structure.
        learned = self._learned.get(self._sig(node), {}).get("selectivity")
        if learned is not None:
            return learned
        return predicate_selectivity(
            node.predicate, self._ndv, self._cfg, self._quantiles, self._mcv
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
            merged.update(self._learned.get("__column_ndv__", {}))  # measured wins
            self._ndv_cache = merged
        return self._ndv_cache

    @property
    def _quantiles(self) -> dict[str, Any]:
        """Learned per-column quantile boundaries
        (`{col: {"probs": [...], "values": [...]}}`, both ascending), used for
        histogram-based range selectivity. Empty until the metadata loop fills it."""
        return self._learned.get("__column_quantiles__", {})

    @property
    def _mcv(self) -> dict[str, dict[str, float]]:
        """Learned per-column most-common-values (`{col: {str(value): frequency}}`),
        used to sharpen equality selectivity on skewed columns to the value's measured
        frequency. Empty until the metadata loop fills it."""
        return self._learned.get("__column_mcv__", {})

    @property
    def _avg_bytes(self) -> dict[str, float]:
        """Learned per-column average byte widths (column name → bytes/row),
        measured from `ColumnStats.avg_byte_width`. Empty until the metadata loop
        fills it; this is what turns the cost model's memory/IO/broadcast axes
        byte-true for wide columns (large strings, embeddings, blob handles)."""
        return self._learned.get("__column_avg_bytes__", {})

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

        For a composite key, the per-key ndvs combine with **exponential backoff**
        (largest at full weight, each subsequent one dampened) rather than a raw
        product: real composite keys are usually correlated, so the independence
        product overshoots. The result lies between `max_k ndv[k]` (the
        perfectly-correlated / functional-dependence floor) and the full product (the
        independence ceiling), capped at the side's rows — a learned ndv reflects the
        *unfiltered* source, but a filtered input can't carry more distinct keys than
        it has rows. Returns `None` when any key lacks a learned distinct count.
        """
        if not keys:
            return None
        ndv = self._ndv
        if not all(k in ndv and ndv[k] > 0 for k in keys):
            return None
        per_key = sorted((ndv[k] for k in keys), reverse=True)
        combined = 1.0
        exponent = 1.0
        for d in per_key:
            combined *= d**exponent
            exponent /= 2.0
        return min(combined, rows)

    def input_sizes(self, node: Join) -> tuple[RelStats, RelStats]:
        """The estimated sizes of a join's two inputs (for build-side choice)."""
        return self.estimate(node.left), self.estimate(node.right)


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
