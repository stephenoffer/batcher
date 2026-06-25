"""The `_Translator` skeleton plus the public `sql()` entry point.

The translator is one stateful class (`_Translator`); its method bodies are
grouped by theme into sibling modules (`clauses`, `scalar`, `subquery`,
`windowing`, `grouping`, `literals`) as free functions that take the translator
instance as their first argument. The methods here are thin delegators so the
class reads as one cohesive object while each theme stays under the file ceiling.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._sql.parser import clauses, grouping, scalar, subquery, windowing
from batcher._sql.parser.core_utils import _alias_of, _has_aggregate
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow
from batcher.plan.expr_ir import AggExpr, Expr

__all__ = ["sql", "translate_ast"]


def sql(
    query: str,
    *,
    dialect: str = "duckdb",
    functions: dict[str, Any] | None = None,
    **tables: Dataset | pa.Table,
) -> Dataset:
    """Parse `query` in `dialect` and translate it against the named tables/functions."""
    import sqlglot

    ast = sqlglot.parse_one(query, read=dialect)
    return translate_ast(ast, functions=functions, **tables)


def translate_ast(
    ast: Any, *, functions: dict[str, Any] | None = None, **tables: Dataset | pa.Table
) -> Dataset:
    """Translate an already-parsed sqlglot statement into a lazy `Dataset`.

    The string entry point (`sql`) and the session DDL path (which has parsed the
    statement to dispatch ``CREATE``/``DROP``) share this one translator entry.
    """
    registry = {name: _as_dataset(t) for name, t in tables.items()}
    return _Translator(registry, functions or {}).statement(ast)


def _as_dataset(t: Dataset | pa.Table) -> Dataset:
    if isinstance(t, Dataset):
        return t
    if isinstance(t, pa.Table):
        return from_arrow(t)
    raise TypeError(f"table must be a Dataset or pyarrow.Table, got {type(t).__name__}")


class _Translator:
    def __init__(self, registry: dict[str, Dataset], functions: dict[str, Any]) -> None:
        self._registry = registry
        self._functions = functions
        self._agg_map: dict[str, tuple[str, AggExpr]] | None = None
        self._agg_n = 0
        self._scalar_sub_n = 0
        self._udf_n = 0

    # --- statement ---------------------------------------------------------
    def statement(self, node) -> Dataset:
        """Translate a top-level statement: a SELECT or a set operation."""
        from sqlglot import expressions as exp

        # WITH name AS (...), ... — translate each CTE in order and register it
        # under its alias so later FROM references resolve. CTEs may reference
        # earlier ones (they are translated and registered sequentially).
        with_ = node.args.get("with") or node.args.get("with_")
        if with_ is not None:
            for cte in with_.expressions:
                self._registry[cte.alias] = self.statement(cte.this)
            # Strip the WITH so the body translates as an ordinary statement.
            node = node.copy()
            node.set("with", None)
            node.set("with_", None)

        if isinstance(node, exp.Union):
            left = self.statement(node.this)
            right = self.statement(node.expression)
            # sqlglot: distinct=True for `UNION`, False for `UNION ALL`.
            distinct = bool(node.args.get("distinct"))
            return left.union(right, distinct=distinct)
        if isinstance(node, exp.Intersect):
            return self.statement(node.this).intersect(self.statement(node.expression))
        if isinstance(node, exp.Except):
            return self.statement(node.this).except_(self.statement(node.expression))
        if isinstance(node, exp.Select):
            return self.select(node)
        raise NotImplementedError(
            f"only SELECT / UNION / INTERSECT / EXCEPT statements are supported, "
            f"got {type(node).__name__}"
        )

    # --- clause building (clauses.py) --------------------------------------
    def select(self, node) -> Dataset:
        return clauses._select(self, node)

    def _from(self, node) -> Dataset:
        return clauses._from(self, node)

    def _table(self, node) -> Dataset:
        return clauses._table(self, node)

    def _order(self, ds: Dataset, order, projections=None) -> Dataset:
        return clauses._order(self, ds, order, projections)

    # --- registered Python functions (udf.py) ------------------------------
    def _hoist_udfs(self, ds: Dataset, clause_nodes):
        from batcher._sql.parser import udf

        return udf._hoist_udfs(self, ds, clause_nodes)

    # --- subquery decorrelation (subquery.py) ------------------------------
    def _apply_subquery_predicates(self, ds: Dataset, pred):
        return subquery._apply_subquery_predicates(self, ds, pred)

    def _decorrelate_scalar_subqueries(self, ds: Dataset, roots) -> Dataset:
        return subquery._decorrelate_scalar_subqueries(self, ds, roots)

    def _reject_correlated(self, select_node) -> None:
        subquery._reject_correlated(select_node)

    def _rewrite_self_joins(self, select_node) -> None:
        subquery._rewrite_self_joins(self, select_node)

    # --- grouping / aggregation (grouping.py) ------------------------------
    def _grouping_sets_union(self, node, group) -> Dataset:
        return grouping._grouping_sets_union(self, node, group)

    def _projection_map(self, ds: Dataset, projections) -> dict[str, Expr]:
        return grouping._projection_map(self, ds, projections)

    def _aggregate(self, ds: Dataset, projections, group, having) -> Dataset:
        return grouping._aggregate(self, ds, projections, group, having)

    # --- window functions (windowing.py) -----------------------------------
    def _is_window(self, p) -> bool:
        return windowing._is_window(p)

    def _inline_named_windows(self, node) -> None:
        windowing._inline_named_windows(node)

    def _window(self, ds: Dataset, projections) -> Dataset:
        return windowing._window(ds, projections)

    # --- scalar expressions (scalar.py) ------------------------------------
    def _scalar(self, node) -> Expr:
        return scalar._scalar(self, node)

    # --- shared AST helpers (core_utils.py) --------------------------------
    def _has_aggregate(self, node) -> bool:
        return _has_aggregate(node)

    def _alias_of(self, p) -> str:
        return _alias_of(p)
