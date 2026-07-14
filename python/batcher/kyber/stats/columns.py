"""Per-operator column-statistics propagation.

Alongside row counts, the estimator carries per-column `ColumnStat`
(min/max/null_count/ndv) through the plan so aggregate, pruning, and existence
shortcuts can be answered from metadata. The cardinal rule is provenance
discipline: an operator may carry a column's *values* forward only as strongly
as it can still vouch for them. A `Sort` preserves the exact value set
(`EXACT` survives); a `Filter` or `Limit` keeps min/max as valid *bounds* but
must downgrade away from `EXACT` because it may have dropped the extremes. The
`weakest`/`downgrade` combiners in `plan.stats` are the only way provenance ever
changes, so nothing can silently over-claim.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from batcher.plan.expr_ir import Cast, Col, Lit
from batcher.plan.logical import Aggregate, Join, Projection
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat, Provenance, RelStats, weakest
from batcher.plan.types import DTYPE_REGISTRY

__all__ = [
    "distinct_columns",
    "filter_columns",
    "global_aggregate_columns",
    "grouped_aggregate_columns",
    "join_columns",
    "limit_columns",
    "project_columns",
    "sample_columns",
    "scan_columns",
    "union_columns",
    "window_columns",
]


def scan_columns(
    source_columns: Mapping[str, ColumnStat],
    learned: Mapping[str, ColumnStat],
) -> dict[str, ColumnStat]:
    """Seed a `Scan`'s column stats from the connector's declared statistics,
    supplemented by the statistics measured for **this source** in past runs.

    Source-declared stats are authoritative and carry their own provenance (footer min/max
    is `EXACT`, byte-truncated string bounds weaker). The learned bundle — an HLL distinct
    count, a KLL quantile grid, Misra-Gries top values, a measured byte width, none of them
    exact — fills in around them.

    The `ndv` merge is the one that matters. A Parquet footer gives EXACT min/max and null
    counts but **never a distinct count**, so the only ndv such a column can have is a
    measured (HLL) one. This used to refuse it — because a `ColumnStat` carried one provenance
    for the whole bundle, so attaching an approximate ndv to an EXACT column would have tagged
    it EXACT and let it answer `count_distinct`. The cost of that refusal was severe: every
    Parquet column reached the optimizer with no ndv, join cardinality fell back to
    `max(|L|, |R|)`, and join ordering went blind (TPC-H q9 applied its most selective filter
    last). `ColumnStat.ndv_provenance` now carries the ndv's *own* tag, so a measured count
    rides alongside exact bounds while `ndv_is_exact` still refuses to answer from it.

    The descriptive statistics (quantiles, mcv, avg_bytes) never carried that risk — nothing
    answers a query from them — so they attach to any column, exact or not.
    """
    cols: dict[str, ColumnStat] = dict(source_columns)
    for name, measured in learned.items():
        existing = cols.get(name)
        if existing is None:
            cols[name] = measured
            continue
        take_ndv = measured.ndv is not None and existing.ndv is None
        cols[name] = dataclasses.replace(
            existing,
            ndv=measured.ndv if take_ndv else existing.ndv,
            # A measured ndv is a sketch, whatever the bundle's bounds are worth.
            ndv_provenance=Provenance.SKETCH if take_ndv else existing.ndv_provenance,
            quantiles=existing.quantiles or measured.quantiles,
            mcv=existing.mcv or measured.mcv,
            avg_bytes=existing.avg_bytes or measured.avg_bytes,
        )
    return cols


def project_columns(
    items: tuple[Projection, ...],
    child: RelStats,
    input_schema: SchemaRef | None = None,
) -> dict[str, ColumnStat]:
    """Project/select output column stats.

    A `col(x)` output carries `x`'s stats through under its alias (exact stays
    exact — projection touches no values); a literal becomes a constant column
    (`min == max == value`, ndv 1, no nulls, `EXACT`); an *identity* `cast` (the
    target type equals the column's own type — the redundant cast the FFI boundary
    already performed on a narrow numeric) is a no-op that carries the source stats
    through unchanged (`input_schema` supplies the source type); any other
    expression is dropped (its output distribution is unknown).
    """
    out: dict[str, ColumnStat] = {}
    for item in items:
        if isinstance(item.expr, Col):
            src = child.columns.get(item.expr.name)
            if src is not None:
                out[item.alias] = src
        elif isinstance(item.expr, Lit):
            value = item.expr.value
            out[item.alias] = ColumnStat(
                min=value, max=value, null_count=0, ndv=1, provenance=Provenance.EXACT
            )
        elif isinstance(item.expr, Cast):
            carried = _identity_cast_column(item.expr, input_schema, child)
            if carried is not None:
                out[item.alias] = carried
    return out


def _identity_cast_column(
    cast: Cast, input_schema: SchemaRef | None, child: RelStats
) -> ColumnStat | None:
    """The carried stat for a `Cast(Col(x), T)` that is a provable no-op, else None.

    Only an *identity* cast — `T` equal to `x`'s own (post-widening) type — is
    treated as value-preserving: it changes nothing, so `x`'s full `ColumnStat`
    (including `EXACT` min/max/ndv) carries through under the output alias. Any
    value-changing cast (a widening/narrowing/opaque conversion, or a `try_cast`
    that can turn a failed conversion into a null) is dropped — its output
    distribution can no longer be vouched for from the input's stats.
    """
    if cast.try_cast or not isinstance(cast.input, Col):
        return None
    if input_schema is None or not input_schema.has(cast.input.name):
        return None
    target = DTYPE_REGISTRY.get(cast.dtype)
    if target is None or not input_schema.field(cast.input.name).type.equals(target):
        return None
    return child.columns.get(cast.input.name)


def filter_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Filter output column stats: min/max survive as *bounds* (a filter can only
    shrink the value range), but provenance drops to `DEFAULT` because the
    extremes may have been removed. null_count becomes unknown; ndv is an upper bound.

    The measured distributional stats (quantiles, mcv, avg_bytes) and the totals
    (total_sum, mean) carry through by `downgrade`, which weakens provenance but keeps the
    values. They must: they describe the column's *shape*, they are read only to estimate,
    and a relation above a filter is exactly where an estimate is still needed — dropping
    them here would mean any predicate at all blinded every operator above it to the very
    statistics the metadata loop measured (a `WHERE` clause would cost a wide string column
    at the flat 64-byte default and forfeit its broadcast join).
    """
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        out[name] = dataclasses.replace(
            stat.downgrade(Provenance.DEFAULT),
            null_count=None,  # filter may drop nulls; count no longer known
        )
    return out


def limit_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Limit output column stats: like a filter, min/max are retained as bounds
    but downgraded (a prefix of rows may exclude the extremes)."""
    return filter_columns(child)


def sample_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Sample output column stats: a sample is a row-shrinking operator, so min/max
    survive only as *bounds* and are downgraded from `EXACT` (the sampled subset may
    drop the extremes), exactly like a filter/limit."""
    return filter_columns(child)


def window_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Window output column stats: a `Window` is strictly row-count preserving and
    only *appends* function columns, so every input column's stats carry through
    unchanged — `EXACT` survives (no value is added, removed, or reordered). The
    appended function columns are left unknown (omitted)."""
    return dict(child.columns)


def distinct_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Distinct output column stats: dedup preserves the exact *value set*, so
    min/max/ndv pass through at their original provenance; null_count is no
    longer known (dedup collapses duplicate nulls to one).

    The measured width carries through (dedup drops rows, it does not change how wide a
    value is), and so does the quantile grid as a description of the value *set*. The
    most-common-values do not: dedup collapses every duplicate, which is precisely the
    frequency information an MCV records — a value held by 90% of rows is one row after a
    `DISTINCT`, so carrying its frequency forward would claim a skew that no longer exists.
    """
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        out[name] = dataclasses.replace(
            stat,
            null_count=None,
            total_sum=None,  # sum changes under dedup
            mean=None,  # so does the mean
            mcv=None,  # dedup destroys frequency; see above
        )
    return out


def union_columns(children: list[RelStats], output_names: list[str]) -> dict[str, ColumnStat]:
    """UNION ALL output column stats, by output position.

    For each output column, min = min over branches, max = max over branches;
    null_count = sum over branches. Each is `EXACT` only when every branch's
    corresponding stat is `EXACT` and present. ndv is left unknown (distinct
    values may overlap across branches).
    """
    out: dict[str, ColumnStat] = {}
    # Resolve each branch's columns in its own output order, aligned by position.
    branch_cols = [list(c.columns.values()) for c in children]
    for pos, name in enumerate(output_names):
        stats: list[ColumnStat] = []
        for bi, branch in enumerate(children):
            # Prefer name match; fall back to positional alignment across branches.
            if name in branch.columns:
                stats.append(branch.columns[name])
            elif pos < len(branch_cols[bi]):
                stats.append(branch_cols[bi][pos])
        if len(stats) != len(children):
            continue  # a branch lacks this column → can't combine safely
        out[name] = _merge_union_column(stats)
    return out


def _merge_union_column(stats: list[ColumnStat]) -> ColumnStat:
    prov = weakest(*(s.provenance for s in stats))
    mins = [s.min for s in stats if s.min is not None]
    maxs = [s.max for s in stats if s.max is not None]
    nulls = [s.null_count for s in stats]
    widths = [s.avg_bytes for s in stats if s.avg_bytes is not None]
    all_min = len(mins) == len(stats)
    all_max = len(maxs) == len(stats)
    all_null = all(n is not None for n in nulls)
    return ColumnStat(
        min=_safe_min(mins) if all_min else None,
        max=_safe_max(maxs) if all_max else None,
        null_count=sum(n for n in nulls if n is not None) if all_null else None,
        ndv=None,
        provenance=prov,
        # A width is a per-row property, so the union's is the widest branch's — the safe
        # direction for the memory/broadcast sizing it feeds. Quantiles and MCVs describe a
        # *distribution* and do not merge by any sound rule here, so they are dropped.
        avg_bytes=max(widths) if widths else None,
    )


def global_aggregate_columns(node: Aggregate, child: RelStats) -> dict[str, ColumnStat]:
    """Derive the one output row's column stats for a *global* (no-key) aggregate.

    Each output alias becomes a constant column whose `min == max == <value>`
    when that aggregate is derivable from the child's exact stats:

      - `count(*)`            = child.rows                  (needs exact rows)
      - `count(col)`          = child.rows - null_count(col)(needs exact rows + null_count)
      - `min(col)` / `max(col)` = col.min / col.max         (needs exact col bound)
      - `sum(col)`            = col.total_sum               (needs a recorded sum)
      - `count_distinct(col)` = col.ndv                     (needs *exact* ndv only)
      - `bool_and`/`bool_or`  = (col.min is True)/(col.max is True)  (boolean col, exact min/max)

    Anything not exactly derivable is omitted, so a downstream reader sees only
    answerable aggregates (provenance `EXACT`).
    """
    out: dict[str, ColumnStat] = {}
    for spec in node.aggregates:
        value = _derive_scalar_aggregate(spec.agg.func, spec.agg.input, child, node.input)
        if value is not None:
            out[spec.alias] = ColumnStat(
                min=value, max=value, null_count=0, ndv=1, provenance=Provenance.EXACT
            )
    return out


def _is_float_column(col_name: str, plan) -> bool:
    """Whether `col_name` is a floating-point column in `plan`'s leaf schema."""
    import pyarrow as pa

    from batcher.plan.logical import Scan
    from batcher.plan.visitor import walk

    for node in walk(plan):
        if isinstance(node, Scan) and node.schema.has(col_name):
            return pa.types.is_floating(node.schema.field(col_name).type)
    return False


# SQL/DataFrame spellings of the two boolean aggregates (`every`/`some` are the
# ANSI-SQL aliases of `bool_and`/`bool_or`). All resolve to the same min/max derivation.
_BOOL_AND_FUNCS = frozenset({"bool_and", "every"})
_BOOL_OR_FUNCS = frozenset({"bool_or", "some"})


def _derive_scalar_aggregate(func: str, input_expr, child: RelStats, plan=None):
    """The exact scalar value of one global aggregate, or None if not derivable.

    `plan` is the aggregate's input, used only to type-check a `min`/`max` column — see the
    float-bound refusal below. It is optional so existing callers/tests keep working; without
    it the float check cannot run.
    """
    col_name = input_expr.name if isinstance(input_expr, Col) else None
    if func == "count_star":
        # count(*) = total rows, regardless of any column.
        return int(child.rows) if child.rows_exact else None
    if func == "count":
        # count(col) = rows - nulls(col); needs exact rows and an exact null count.
        # When null_count is a known 0 this is exactly `child.rows` — the common
        # "count a non-null column" case answered without any per-row work.
        if not child.rows_exact or col_name is None:
            return None
        stat = child.columns.get(col_name)
        if stat is None or stat.provenance is not Provenance.EXACT or stat.null_count is None:
            return None
        return int(child.rows - stat.null_count)
    if col_name is None:
        return None
    stat = child.columns.get(col_name)
    if stat is None or stat.provenance is not Provenance.EXACT:
        return None
    if func == "min":
        # Sound for every type, floats included: NaN is the *greatest* value in the total
        # order (see `max` below), so a dropped NaN can never have been the minimum. An
        # all-NaN column has no bound at all (the sketch saw nothing, the footer has no
        # min/max), so it falls through to execution rather than answering NULL.
        return stat.min
    if func == "max":
        # A float bound cannot answer `max`. Both producers of the bound deliberately drop
        # NaN — the KLL quantile sketch ignores it on `add` ("no place in an ordered sketch")
        # and the Parquet spec omits it from column statistics — while SQL's total order (the
        # one our own ORDER BY uses) makes NaN the *greatest* value. So `max(f)` over a column
        # containing a NaN **is** NaN, which the bound cannot represent, and nothing in the
        # stats records whether a NaN was dropped. Answering from the bound returned the
        # largest non-NaN value and silently disagreed with executing the very same query —
        # a wrong answer produced *by an optimization*, which is the worst kind. Execute.
        #
        # Only `max`, and only floats: `min` is unaffected (above), and ints / strings /
        # temporals have no NaN, so they keep answering from metadata.
        if plan is not None and col_name is not None and _is_float_column(col_name, plan):
            return None
        return stat.max
    if func == "sum":
        # An exact recorded total (a catalog/format `total_sum`). SQL `sum` over a
        # group with no non-null value is NULL, so a provably-empty relation is not
        # derivable — return None to fall back rather than claim a spurious 0.
        if child.rows_exact and child.rows == 0:
            return None
        return stat.total_sum
    if func == "mean":
        # SQL `avg`/`mean` of the non-null values (a recorded exact mean). NULL over an
        # all-null / empty group — not derivable, fall back rather than divide by zero.
        return stat.mean
    if func == "count_distinct":
        # SQL `count(distinct col)` excludes NULL; the `ndv` contract is likewise the number
        # of distinct *non-null* values, so an EXACT ndv is the answer directly. The gate is
        # `ndv_is_exact`, not the bundle's provenance: a Parquet column now carries a measured
        # (SKETCH) ndv alongside its exact bounds, and that measured count must never answer
        # an exact `count_distinct` — it exists only to inform cost and cardinality.
        if stat.ndv is None or not stat.ndv_is_exact:
            return None
        return int(stat.ndv)
    if func in _BOOL_AND_FUNCS or func in _BOOL_OR_FUNCS:
        return _derive_bool_aggregate(func, stat, child)
    return None


def _derive_bool_aggregate(func: str, stat: ColumnStat, child: RelStats):
    """`bool_and`/`bool_or` (SQL `every`/`some`) of a boolean column from EXACT min/max.

    Over a boolean column, `bool_and` is true iff *every* non-null value is true —
    exactly `min == True` (False sorts below True); `bool_or` is true iff *any* is —
    exactly `max == True`. Both ignore NULLs (SQL semantics), and the footer min/max
    already exclude nulls. An all-null or empty group has no min/max to derive from
    and SQL returns NULL there, so this returns None (fall back) in that case.
    """
    if not isinstance(stat.min, bool) or not isinstance(stat.max, bool):
        return None  # not a boolean column (or no non-null values recorded)
    # An all-null / empty group returns SQL NULL — not derivable as an exact bool.
    if child.rows_exact and stat.null_count is not None and child.rows - stat.null_count <= 0:
        return None
    return bool(stat.min) if func in _BOOL_AND_FUNCS else bool(stat.max)


def grouped_aggregate_columns(node: Aggregate, child: RelStats) -> dict[str, ColumnStat]:
    """Column stats for a *grouped* aggregate's GROUP BY key outputs.

    A bare-`Col` group key appears verbatim in the output, holding exactly the set
    of *distinct* key values of the input. Grouping invents no value and drops no
    extreme, so the key column's `min`/`max` carry through as **EXACT** bounds at the
    child's provenance (like `Distinct`). `null_count` is dropped (duplicate nulls
    collapse to one group) and `ndv` is not claimed here: the number of groups is only
    an *estimate*, so tagging it EXACT would let `count_distinct` answer from a guess.
    Per-group aggregate outputs are not constant, so only the keys are derived.
    """
    out: dict[str, ColumnStat] = {}
    for key in node.group_keys:
        if isinstance(key.expr, Col):
            src = child.columns.get(key.expr.name)
            if src is not None:
                out[key.alias] = ColumnStat(
                    min=src.min,
                    max=src.max,
                    null_count=None,
                    ndv=None,
                    provenance=src.provenance,  # extremes preserved → EXACT survives
                    bloom=src.bloom,  # grouping adds no value → absence proof holds
                    # A key's width is unchanged by grouping; its frequency distribution is
                    # destroyed by it (every group is one row), so the mcv must not carry.
                    avg_bytes=src.avg_bytes,
                    quantiles=src.quantiles,
                )
    return out


def join_columns(
    node: Join, left: RelStats, right: RelStats, out_rows: float | None = None
) -> dict[str, ColumnStat]:
    """Column stats for a join's output: each side's values as downgraded *bounds*.

    A join only removes rows from a side or repeats them (an FK match), never invents
    a new value, so a preserved column's `min`/`max` stay valid *bounds* — but the
    extremes may be dropped and provenance must fall from `EXACT` (a join is a
    row-shrinking/duplicating operator). `null_count` is dropped (a match may duplicate
    rows or an outer join add nulls). The membership bloom survives — a value absent
    from a side stays absent in any join output of it.

    `ndv` is carried forward as `min(ndv_in, out_rows)`. Because a join invents no
    values, the output's distinct count for a preserved column can only *fall* (rows
    are dropped) — never rise — and it is trivially bounded by the output row count;
    the minimum of the two is therefore a sound upper bound and the standard Selinger
    estimate. Dropping it instead (the previous behaviour) left every join *above* a
    join with no key NDV, so `_estimate_join` fell back to `max(|L|, |R|)` and could
    not see that an upstream selective join shrinks the pipeline — which is what
    steered TPC-H Q9 into multi-gigabyte `lineitem ⋈ partsupp ⋈ orders` intermediates
    before joining the 5%-selective `part`. `out_rows` is the join's estimated output
    cardinality; when unknown, `ndv` is dropped as before.
    """
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        side = left if o.side == "left" else right
        src = side.columns.get(o.name)
        if src is not None:
            out[o.alias] = dataclasses.replace(
                src.downgrade(Provenance.DEFAULT),
                null_count=None,
                ndv=_join_ndv(src.ndv, out_rows),
                total_sum=None,  # matching duplicates rows, so a recorded sum no longer holds
                mean=None,
                # The measured *width* survives — a join changes which rows are present, not
                # how many bytes one holds — and it is what keeps byte-true memory and
                # broadcast sizing alive above a join. Frequencies (mcv) do not: a join
                # re-weights the value distribution by its match multiplicity.
                mcv=None,
            )
    return out


def _join_ndv(ndv: float | None, out_rows: float | None) -> float | None:
    """A preserved column's distinct count after a join: `min(ndv, out_rows)`."""
    if ndv is None or out_rows is None:
        return None
    return max(1.0, min(ndv, out_rows))


def _safe_min(values: list):
    try:
        return min(values)
    except TypeError:
        return None


def _safe_max(values: list):
    try:
        return max(values)
    except TypeError:
        return None
