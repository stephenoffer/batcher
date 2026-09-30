"""Multi-level grouped aggregation — `ROLLUP`, `CUBE` and `GROUPING SETS`.

The SQL front-end has had these since it could parse them; the DataFrame surface had
no spelling for them at all, so a subtotal report had to be written as a `union` of
hand-written `group_by`s (or in SQL). This module is that spelling:
`ds.rollup("region", "city").agg(...)`.

A multi-level GROUP BY is **not** a distinct execution strategy here — the same choice
the SQL translator makes, for the same reason. Each level is an ordinary `group_by`
over its active keys, with the inactive keys grouped by a *typed null*
(`nullif(col, col)`, a null of the column's own type, which also keeps one row per
level), and the levels are stacked with `union(distinct=False)`. So every level is a
plan the optimizer, the spill path and the distributed executor already understand,
and nothing in the aggregate path needs to know that levels exist.

**The levels share one aggregate, not just one input.** Every level recomputing the input
cost TPC-DS q22 its 11.7M-row inventory join five times, and plan-level common-subplan reuse
declined to share it: the join's *estimated* result (16.5M rows, against 2.3M actual) was
over its byte cap. The levels do not need the input, only a summary of it: every level's keys
are a subset of the union of all of them, so one aggregate at that finest grouping, holding
each aggregate's *partial* state, determines every level, and a level becomes a second, small
aggregate over it merging partials (a sum of sums, a sum of counts, a min of mins). That
finest aggregate is what reuse then materializes once. It is built here, where the levels are
stacked, rather than as a Kyber rule, because reuse matches repeated subtrees on the plan as
written -- a sharing only the optimizer could see would never be materialized. Only aggregates
with a partial form qualify (`agg_pushdown._PREAGG_MERGE`, plus `mean` as its sum/count pair
over a numeric column); anything else keeps one independent aggregate per level.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.kyber.common_subplan import structural_key
from batcher.kyber.rules.agg_pushdown.rules import _PREAGG_MERGE
from batcher.plan.expr_ir import AggExpr, Col, Expr, coalesce, col, lit, nullif
from batcher.plan.expr_ir.nodes import NullIf
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Join,
    LogicalPlan,
    Project,
    Projection,
    Union,
    share_sources,
)
from batcher.plan.visitor import walk

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["MultiLevelGroupBy", "cube_levels", "rollup_levels", "stack_levels"]


def rollup_levels(keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The grouping levels of `ROLLUP(k₁, …, kₙ)`: every prefix, longest first.

    Args:
        keys: The rollup keys, most significant first.

    Returns:
        The prefixes of `keys` from the full list down to the empty grand total.
    """
    return [keys[:i] for i in range(len(keys), -1, -1)]


def cube_levels(keys: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The grouping levels of `CUBE(k₁, …, kₙ)`: every subset, largest first.

    Args:
        keys: The cube keys.

    Returns:
        Every subset of `keys`, ordered by decreasing size, each in the original
        key order so the output column order is stable.
    """
    return [
        tuple(c) for size in range(len(keys), -1, -1) for c in itertools.combinations(keys, size)
    ]


def stack_levels(frames: Sequence[Dataset]) -> Dataset:
    """Stack one grouping level per frame into a single relation, sharing their sources.

    The levels of a multi-level `GROUP BY` are the same query over the same relations,
    differing only in what each groups by, so they must land on **one** source list.
    `Dataset.union` cannot know that -- it takes arbitrary datasets, so it renumbers each
    one's scans and concatenates the lists -- and going through it binds the same relation
    once per level: 5 levels over TPC-DS q22's three tables is 15 bindings of 3 relations.
    That is not cosmetic. It is what stops plan-level common-subplan reuse recognizing the
    levels as sharing a subtree, so each level re-reads and re-joins the whole input, and
    q22 ran 61x DuckDB.

    This is the one shape whose branches can be *proved* to be the same relations, which is
    why the sharing lives here and not in `Dataset.union` -- that must keep renumbering,
    because its inputs are unrelated in general.

    Both front-ends stack their levels through this function. The SQL translator builds its
    levels by re-translating one SELECT per level, so its frames arrive with a source list
    each; `share_sources` merges them by object identity, which is a no-op for the DataFrame
    path (every level is derived from one `Dataset`, so there is one list already) and the
    whole point for SQL.

    Args:
        frames: One `Dataset` per grouping level, in output order. Every frame must carry
            the same columns, as for `Dataset.union`.

    Returns:
        A `Dataset` concatenating the levels, its scans renumbered onto one source list.
    """
    if len(frames) == 1:
        return frames[0]
    # Local import: `api.dataset.frame` imports this module, so naming `Dataset` at module
    # scope would close the cycle.
    from batcher.api.dataset import Dataset

    plans, sources = share_sources([(f._plan, f._sources) for f in frames])
    union = Union(tuple(plans), False)
    return Dataset(_share_level_input(union) or union, sources)


class MultiLevelGroupBy:
    """An in-progress multi-level aggregation, from `Dataset.rollup`/`cube`/`grouping_sets`.

    Not constructed directly. Like `GroupBy` it is a lazy builder with one finisher,
    `agg`, which returns a `Dataset` whose rows are every level's groups stacked in
    level order — the aggregated levels first, the grand total last.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": ["e", "e", "w"], "c": ["x", "y", "z"], "v": [1, 2, 4]})
            >>> out = ds.rollup("r", "c").agg(total=bt.col("v").sum())
            >>> out.sort("r", "c").to_pydict()["total"]
            [1, 2, 3, 4, 4, 7]
    """

    __slots__ = ("_ds", "_keys", "_levels")

    def __init__(self, ds: Dataset, keys: tuple[str, ...], levels: list[tuple[str, ...]]) -> None:
        """Hold the source dataset, the full key list, and the levels to aggregate."""
        self._ds = ds
        self._keys = keys
        self._levels = levels

    def __repr__(self) -> str:
        """A source-like rendering naming the keys and the level count."""
        return f"MultiLevelGroupBy(keys={list(self._keys)!r}, levels={len(self._levels)})"

    def agg(self, **named: AggExpr | Expr) -> Dataset:
        """Aggregate every grouping level and stack the results.

        Each level's inactive keys read as NULL, which is how SQL marks a subtotal row.
        Use ``grouping(...)`` semantics — testing a key for null — to tell a subtotal
        apart from a genuine null in the data.

        Args:
            **named: Output names bound to aggregate expressions, as for
                :meth:`GroupBy.agg`.

        Returns:
            A new `Dataset` with the key columns followed by the aggregates, one block
            of rows per level.

        Raises:
            PlanError: If no aggregates are given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"r": ["e", "e", "w"], "v": [1, 2, 4]})
                >>> ds.rollup("r").agg(n=bt.col("v").sum()).sort("r").to_pydict()
                {'r': ['e', 'w', None], 'n': [3, 4, 7]}
        """
        if not named:
            raise PlanError(
                "rollup()/cube()/grouping_sets() need at least one aggregate, "
                "e.g. .agg(total=col('x').sum())"
            )
        frames = [self._level_frame(level, named) for level in self._levels]
        assert frames  # `_levels` is never empty: the grand total is always one
        return stack_levels(frames)

    def _level_frame(self, level: tuple[str, ...], named: dict[str, AggExpr | Expr]) -> Dataset:
        """One level: group by its active keys, null the rest, and order the columns.

        The inactive keys are grouped by ``nullif(col, col)`` rather than projected
        afterwards, which does two things at once: the null carries the column's own
        type (so every level's schema matches and the union is legal), and it is a
        constant key, so the level collapses to the groups of its active keys.

        The grand total takes the ungrouped aggregate instead, because a `GROUP BY` on
        constant keys is not the same relation as SQL's `GROUP BY ()`. They agree on
        every non-empty input and disagree on the empty one: grouping yields one group
        per distinct key value, so *no* rows when there are no rows, where the grand
        total is defined to yield exactly one. A rollup whose input a filter happened to
        empty lost its total line with no error. The keys still have to carry their own
        types for the union above to be legal, which `nullif(max(k), max(k))` gives —
        always NULL, typed as `k`, and an aggregate, so it is legal with no GROUP BY.
        """
        active = set(level)
        if not active:
            nulls: dict[str, AggExpr | Expr] = {
                k: nullif(col(k).max(), col(k).max()) for k in self._keys
            }
            return self._ds.agg(**named, **nulls).select(*self._keys, *named)
        keyed = {k: col(k) if k in active else nullif(col(k), col(k)) for k in self._keys}
        grouped = self._ds.group_by(**keyed).agg(**named)
        return grouped.select(*self._keys, *named)


#: Prefix of the shared aggregate's partial columns; also what marks a union already rewritten.
_PARTIAL = "__lvl_"


def _share_level_input(node: Union) -> LogicalPlan | None:
    """`UNION ALL` of aggregates over one input -> one finest aggregate, rolled up per level.

    Applies only when every branch is an aggregate (under an optional projection) over a
    structurally identical input that contains a join or an aggregate -- a cheap input is
    cheaper to re-read than to summarize -- and every group key is a column or a typed null
    of one. Result-invariant: each level merges exact partials of its own aggregates.

    Args:
        node: The stacked levels.

    Returns:
        The union with each level computed from the shared finest aggregate, or None.
    """
    if node.distinct or len(node.inputs) < 2:
        return None
    levels = [_level_aggregate(branch) for branch in node.inputs]
    if any(level is None for level in levels):
        return None
    aggregates: list[Aggregate] = [level for level in levels if level is not None]
    base = aggregates[0].input
    if _is_shared_partial(base) or not any(isinstance(n, Join | Aggregate) for n in walk(base)):
        return None
    base_key = structural_key(base)
    if base_key is None or any(structural_key(a.input) != base_key for a in aggregates[1:]):
        return None

    key_columns: dict[str, None] = {}
    for agg in aggregates:
        for key in agg.group_keys:
            column, active = _key_column(key.expr)
            if column is None:
                return None
            if active:
                key_columns[column] = None

    partials: dict[tuple[str, str], str] = {}
    specs: list[AggregateSpec] = []
    for agg in aggregates:
        for spec in agg.aggregates:
            states = _partials_of(spec.agg, base, key_columns)
            if states is None:
                return None
            for key, partial in states:
                if key not in partials:
                    partials[key] = f"{_PARTIAL}{len(partials)}"
                    specs.append(AggregateSpec(partials[key], partial))
    finest = Aggregate(base, tuple(Projection(c, Col(c)) for c in key_columns), tuple(specs))

    rolled = [
        _roll_up(branch, agg, finest, partials)
        for branch, agg in zip(node.inputs, aggregates, strict=True)
    ]
    return dataclasses.replace(node, inputs=tuple(rolled))


def _level_aggregate(branch: LogicalPlan) -> Aggregate | None:
    """The aggregate a union branch computes, directly or under one projection."""
    if isinstance(branch, Project):
        branch = branch.input
    return branch if isinstance(branch, Aggregate) and branch.watermark is None else None


def _is_shared_partial(plan: LogicalPlan) -> bool:
    """Whether `plan` is already a finest aggregate built here, so levels are not re-shared."""
    return isinstance(plan, Aggregate) and any(
        spec.alias.startswith(_PARTIAL) for spec in plan.aggregates
    )


def _key_column(expr: Expr) -> tuple[str | None, bool]:
    """A group key's column and whether it is active, or `(None, False)` for any other key.

    An active key is the column itself; an inactive one is the typed null
    `nullif(col, col)` the multi-level lowering groups it by.
    """
    if isinstance(expr, Col):
        return expr.name, True
    if (
        isinstance(expr, NullIf)
        and isinstance(expr.left, Col)
        and isinstance(expr.right, Col)
        and expr.left.name == expr.right.name
    ):
        return expr.left.name, False
    return None, False


def _plain(agg: AggExpr) -> bool:
    """An aggregate with no second input, parameter, interpolation or ordering."""
    return agg.input2 is None and agg.param is None and not agg.order_by and not agg.interpolation


def _partials_of(
    agg: AggExpr, base: LogicalPlan, key_columns: dict[str, None]
) -> list[tuple[tuple[str, str], AggExpr]] | None:
    """The partial states `agg` is merged from, keyed by `(function, operand IR)`, or None.

    Two levels asking for the same state share one column of the finest aggregate, which is
    why the key is the operand's wire form rather than the `Expr` object. A `min`/`max` of a
    column the finest aggregate groups by needs no state at all -- the level reads the key
    column itself -- which is the grand total's typed-null placeholder, `max(k)` per key.
    """
    if not _plain(agg):
        return None
    if _reads_key(agg, key_columns):
        return []
    operand = _operand_key(agg)
    if agg.func in _PREAGG_MERGE:
        return [((agg.func, operand), AggExpr(agg.func, agg.input))]
    if agg.func == "mean" and _is_numeric_column(agg.input, base):
        return [
            (("sum", operand), AggExpr("sum", agg.input)),
            (("count", operand), AggExpr("count", agg.input)),
        ]
    return None


def _reads_key(agg: AggExpr, key_columns: dict[str, None]) -> bool:
    """Whether `agg` is a `min`/`max` of one of the finest aggregate's key columns."""
    if agg.func not in ("min", "max") or not isinstance(agg.input, Col):
        return False
    return agg.input.name in key_columns


def _operand_key(agg: AggExpr) -> str:
    """`agg`'s operand as its IR string: equal for equal computations, whatever the object."""
    return "" if agg.input is None else repr(agg.input.to_ir())


def _is_numeric_column(expr: Expr | None, base: LogicalPlan) -> bool:
    """Whether `expr` is a column of an integer or floating type in `base`'s schema.

    `mean` over those is a float64, which the sum over the count reproduces. Other input types
    (decimals above all) are declined rather than assumed to give the same type and rounding.
    """
    if not isinstance(expr, Col):
        return False
    schema = base.available_schema()
    if schema is None or expr.name not in schema.arrow.names:
        return False
    dtype = schema.arrow.field(expr.name).type
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype)


def _roll_up(
    branch: LogicalPlan,
    level: Aggregate,
    finest: Aggregate,
    partials: dict[tuple[str, str], str],
) -> LogicalPlan:
    """`branch` with its aggregate computed from `finest` instead of from the input.

    The level keeps its own keys, which read `finest`'s key columns, and merges each
    aggregate's partials; a projection then restores the aggregate's output names and order,
    so whatever sat above the aggregate reads exactly the columns it read before.
    """
    merged: list[AggregateSpec] = []
    finals: list[Projection] = []
    for i, spec in enumerate(level.aggregates):
        agg = spec.agg
        operand = _operand_key(agg)
        if agg.func == "mean":
            total, count = f"__lvl_s{i}", f"__lvl_c{i}"
            merged.append(AggregateSpec(total, AggExpr("sum", Col(partials[("sum", operand)]))))
            merged.append(AggregateSpec(count, AggExpr("sum", Col(partials[("count", operand)]))))
            expr = Col(total).cast("float64") / Col(count).cast("float64")
        else:
            out = f"__lvl_m{i}"
            # A `min`/`max` of a key column has no partial: it reads the key column itself.
            key = agg.input.name if isinstance(agg.input, Col) else ""
            state = Col(partials.get((agg.func, operand), key))
            merged.append(AggregateSpec(out, AggExpr(_PREAGG_MERGE[agg.func], state)))
            # A count over no rows is 0, and a sum of no partial counts is NULL: the grand
            # total of an empty input is the one level where that difference can show.
            expr = coalesce(Col(out), lit(0)) if agg.func in ("count", "count_star") else Col(out)
        finals.append(Projection(spec.alias, expr))
    keys = tuple(Projection(k.alias, Col(k.alias)) for k in level.group_keys)
    rolled: LogicalPlan = Project(
        Aggregate(finest, level.group_keys, tuple(merged)), (*keys, *finals)
    )
    if isinstance(branch, Project):
        rolled = dataclasses.replace(branch, input=rolled)
    return rolled
