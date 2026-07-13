"""Column-level lineage — which source columns each output column is derived from.

The question a governance team actually asks is not "which tables did this query read"
but "if `customers.ssn` is PII, which columns of my derived table carry it?" That is
column-level lineage, and it is what makes a `tag` more than a label: tag a column once,
and lineage tells you every downstream column the tag must follow.

A pure `LogicalPlan → dict` analysis, computed bottom-up. It reads the plan and executes
nothing, so it costs nothing and works on a `Dataset` that was never run.

**It over-approximates, never under-approximates.** An operator this module does not
recognize — `map_batches`, the opaque Python/ML escape hatch — is treated as though every
output column derives from every input column. For a governance answer that is the only
safe direction to be wrong in: a false "this might carry PII" costs a review, a false
"this cannot" costs a breach.

**It tracks data flow, not control flow.** ``filter(col("ssn") == x)`` does not put `ssn`
in the lineage of the surviving columns, even though the filtered *row set* depends on it.
That matches how Unity Catalog and Snowflake report lineage; a column whose values never
reach the output is not a column the output is derived from.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from batcher.plan.expr_ir import Expr, referenced_columns
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
    Window,
)
from batcher.plan.visitor import children

__all__ = ["Origin", "column_lineage"]

#: A source column: ``(table, column)``.
Origin = tuple[str, str]

#: Output column name → the source columns its values are derived from.
LineageMap = dict[str, frozenset[Origin]]


def column_lineage(plan: LogicalPlan, tables: Sequence[str]) -> LineageMap:
    """Return, for each of `plan`'s output columns, the source columns it derives from.

    Operates on a `LogicalPlan`; `Dataset.lineage` is the sugar over it that names the
    tables for you and renders the origins as ``"table.column"``.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.governance import column_lineage
            >>> from batcher.plan.expr_ir import Col
            >>> from batcher.plan.logical import Project, Projection, Scan
            >>> from batcher.plan.schema import SchemaRef
            >>> schema = SchemaRef.from_arrow(
            ...     pa.schema([("first", pa.string()), ("last", pa.string())])
            ... )
            >>> plan = Project(
            ...     Scan(0, schema),
            ...     (Projection(alias="name", expr=Col("first") + Col("last")),),
            ... )
            >>> sorted(column_lineage(plan, ["people.parquet"])["name"])
            [('people.parquet', 'first'), ('people.parquet', 'last')]

    Args:
        plan: The plan to analyze. It is not executed.
        tables: The table name of each source, indexed by a `Scan`'s ``source_id``.

    Returns:
        A mapping from output column name to the set of ``(table, column)`` origins whose
        *values* flow into it. A column built only from literals (``lit(1)``) or generated
        (``with_row_index``) maps to the empty set — it has no origin.
    """
    return _lineage(plan, tables)


def _union(sets: Iterable[frozenset[Origin]]) -> frozenset[Origin]:
    """Union of origin sets (empty when there are none)."""
    out: frozenset[Origin] = frozenset()
    for s in sets:
        out |= s
    return out


def _from(child: LineageMap, expr: Expr) -> frozenset[Origin]:
    """The origins of everything `expr` reads."""
    return _union(child.get(c, frozenset()) for c in referenced_columns(expr))


def _lineage(node: LogicalPlan, tables: Sequence[str]) -> LineageMap:
    if isinstance(node, Scan):
        table = tables[node.source_id] if node.source_id < len(tables) else ""
        return {c: frozenset({(table, c)}) for c in node.available_columns()}

    if isinstance(node, Project):
        child = _lineage(node.input, tables)
        return {item.alias: _from(child, item.expr) for item in node.items}

    if isinstance(node, Filter | Limit | Sort | Distinct | Sample):
        # Row-set operators: they choose *which* rows survive, never what a value is.
        return _lineage(node.input, tables)

    if isinstance(node, Aggregate):
        child = _lineage(node.input, tables)
        out = {key.alias: _from(child, key.expr) for key in node.group_keys}
        for spec in node.aggregates:
            origins: frozenset[Origin] = frozenset()
            for arg in (spec.agg.input, spec.agg.input2):
                if arg is not None:
                    origins |= _from(child, arg)
            out[spec.alias] = origins  # `count()` has no input → no origin
        return out

    if isinstance(node, Window):
        child = _lineage(node.input, tables)
        # A window value depends on its own input *and* on how the rows were partitioned
        # and ordered — `rank()` over `salary` is derived from `salary` despite taking no
        # argument, and reveals its ordering.
        frame: frozenset[Origin] = frozenset()
        for key in node.partition_keys:
            frame |= _from(child, key)
        for key in node.order_keys:
            frame |= _from(child, key.expr)
        out = dict(child)  # window preserves every input column
        for fn in node.functions:
            out[fn.alias] = frame | (
                _from(child, fn.input) if fn.input is not None else frozenset()
            )
        return out

    if isinstance(node, Join):
        left, right = _lineage(node.left, tables), _lineage(node.right, tables)
        sides = {"left": left, "right": right}
        return {col.alias: sides[col.side].get(col.name, frozenset()) for col in node.output}

    if isinstance(node, Union):
        # Positional: the i-th output column is fed by the i-th column of every input.
        per_input = [_lineage(inp, tables) for inp in node.inputs]
        names = [list(inp.available_columns()) for inp in node.inputs]
        return {
            alias: _union(
                lin.get(cols[i], frozenset()) for lin, cols in zip(per_input, names, strict=True)
            )
            for i, alias in enumerate(names[0])
        }

    if isinstance(node, RowId):
        return {**_lineage(node.input, tables), node.alias: frozenset()}  # generated

    if isinstance(node, Unnest):
        child = _lineage(node.input, tables)
        out = {c: child[c] for c in child if c != node.column}
        out[node.alias] = child.get(node.column, frozenset())
        return out

    if isinstance(node, Unpivot):
        child = _lineage(node.input, tables)
        melted = _union(child.get(c, frozenset()) for c in node.on)
        out = {c: child.get(c, frozenset()) for c in node.index}
        out[node.variable_name] = melted  # the column *names* come from the melted columns
        out[node.value_name] = melted
        return out

    # An operator this analysis does not model — `map_batches` and anything added later.
    # Every output column is assumed to derive from every input column. Over-approximating
    # is the only safe direction: a false positive costs a review, a false negative a leak.
    child_lineage: LineageMap = {}
    for child_node in children(node):
        child_lineage |= _lineage(child_node, tables)
    everything = _union(child_lineage.values())
    return dict.fromkeys(node.available_columns(), everything)
