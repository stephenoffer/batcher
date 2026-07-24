"""Aggregate output column statistics — the values a grouped/global aggregate produces.

Split from `columns` (which propagates a row-preserving operator's column stats) because
deriving an aggregate *output* is a different job: a global aggregate's outputs are provable
*constants* when the child's exact stats determine them (`count(*)`, `min`/`max`,
`count_distinct`, the boolean aggregates), and a grouped aggregate's group-key outputs carry
the key column's bounds forward. Kept here as one cohesive family.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Aggregate
from batcher.plan.stats import ColumnStat, Provenance, RelStats

__all__ = ["global_aggregate_columns", "grouped_aggregate_columns"]


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
        # The null count's own tag, not the bundle's: a string column's bounds may be truncated
        # (so the bundle is DEFAULT) while its footer null count is exact — and `count(name)` is
        # derived from the null count, not from the bounds.
        if stat is None or stat.null_count is None or not stat.null_count_is_exact:
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
    out.update(_grouped_aggregate_bounds(node, child))
    return out


# Aggregates whose per-group value is always one of the input column's own values, or lies
# between them: `min`/`max` return an actual value, and an average of values inside `[lo, hi]`
# is inside `[lo, hi]`. So the child column's bounds bound the aggregate's output too.
_ORDER_BOUNDED_AGGS = frozenset({"min", "max", "mean", "avg", "median"})
_COUNTING_AGGS = frozenset({"count", "count_star", "count_distinct"})


def _grouped_aggregate_bounds(node: Aggregate, child: RelStats) -> dict[str, ColumnStat]:
    """Bounds for a grouped aggregate's *value* outputs — not constants, but not unknown.

    A grouped aggregate's outputs vary by group, so none of them is the provable constant a
    global aggregate produces. That is not the same as knowing nothing about them, and the
    difference matters because a `HAVING` clause (or any filter above the aggregate) is a
    predicate on exactly these columns. With no statistics at all it falls to a flat constant;
    with bounds it can interpolate, and a `HAVING max(v) > 10^6` over a column whose maximum
    is 1,000 is *provably* empty without executing anything.

    Two families are bounded without any assumption:

    * **order-bounded** — `min`/`max`/`avg`/`median` of a column return a value inside that
      column's own `[min, max]`, whatever the grouping is;
    * **counting** — a group has at least one row and at most all of them, so a per-group
      count lies in `[1, |child|]` — but **only when `|child|` is EXACT**. `min`/`max` stay
      valid *bounds* under a downgraded provenance (see `ColumnStat`: a filtered column may
      no longer attain its extremes, yet they still enclose it), and the counting upper
      bound is the one place that does not hold: an *estimated* `|child|` can be smaller
      than the truth, and a too-small upper bound is not conservative, it is wrong.

      `zonemap_prune_filter` folds a `HAVING count(*) > n` whose bound cannot reach `n`
      into the empty relation — deleting every row. Publishing an estimate here therefore
      turned a learned row count into missing results: after a selective query taught the
      hub that a scan yields ~1 row, an unrelated ``GROUP BY s HAVING count(*) > 6`` over
      the *same source with a different predicate* returned nothing, and a later, less
      selective query silently restored the right answer. An estimate may choose a plan; it
      may never decide which rows exist.

    Always `DEFAULT` provenance: these are bounds on a *set* of values, so nothing here may
    answer an exact terminal — and a group's actual extreme is generally strictly inside them.
    """
    out: dict[str, ColumnStat] = {}
    for spec in node.aggregates:
        func = spec.agg.func
        if func in _COUNTING_AGGS:
            exact_rows = child.provenance.is_exact and child.rows > 0
            hi = child.rows if exact_rows else None
            out[spec.alias] = ColumnStat(
                min=1 if func != "count" else 0,  # count(col) is 0 for an all-null group
                max=int(hi) if hi is not None else None,
                null_count=0,
                provenance=Provenance.DEFAULT,
            )
            continue
        if func not in _ORDER_BOUNDED_AGGS or not isinstance(spec.agg.input, Col):
            continue
        src = child.columns.get(spec.agg.input.name)
        if src is None or src.min is None or src.max is None:
            continue
        out[spec.alias] = ColumnStat(
            min=src.min, max=src.max, provenance=Provenance.DEFAULT, avg_bytes=src.avg_bytes
        )
    return out
