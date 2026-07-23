"""The `_Translator` skeleton plus the public `sql()` entry point.

The translator is one stateful class (`_Translator`); its method bodies are
grouped by theme into sibling modules (`clauses`, `subquery`, `windowing`,
`grouping`, and the `expressions` subpackage) as free functions that take the
translator instance as their first argument. The methods here are thin delegators so the
class reads as one cohesive object while each theme stays under the file ceiling.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import parse_sql
from batcher._sql.parser import (
    clauses,
    expressions,
    from_clause,
    grouping,
    grouping_sets,
    subquery,
    windowing,
)
from batcher._sql.parser.core_utils import _alias_of, _disambiguate_columns, _has_aggregate
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
    """Parse `query` in `dialect` and translate it against the named tables/functions.

    A parse failure is re-raised as `PlanError` with a plain-text message; see
    `batcher._internal.sql_errors.parse_sql`.
    """
    return translate_ast(parse_sql(query, dialect=dialect), functions=functions, **tables)


def translate_ast(
    ast: Any, *, functions: dict[str, Any] | None = None, **tables: Dataset | pa.Table
) -> Dataset:
    """Translate an already-parsed sqlglot statement into a lazy `Dataset`.

    The string entry point (`sql`) and the session DDL path (which has parsed the
    statement to dispatch ``CREATE``/``DROP``) share this one translator entry.
    """
    registry = {name: _as_dataset(t) for name, t in tables.items()}
    return _Translator(registry, functions or {}).statement(ast)


# Iteration cap for a recursive CTE. A wrong or missing stop condition would otherwise
# loop forever; failing loudly at a generous bound is better than hanging. DuckDB's
# equivalent guard is 1024 by default.
_MAX_RECURSION = 1024


def _references(node, name: str) -> bool:
    """Whether `node` reads the table `name` anywhere beneath it."""
    from sqlglot import expressions as exp

    return any(t.name == name for t in node.find_all(exp.Table))


def _is_self_referential(cte) -> bool:
    """Whether a CTE's body references the CTE itself — i.e. it is the recursive one.

    `WITH RECURSIVE` marks the whole `WITH` clause, not the individual CTEs, so a recursive
    block may still hold ordinary non-recursive CTEs alongside the recursive one. Only the
    self-referential ones need fixpoint evaluation.
    """
    return _references(cte.this, cte.alias)


def _table_ref_count(root, name: str) -> int:
    """How many times `name` is referenced as a table anywhere under `root`.

    Counts `FROM name` / `JOIN name` occurrences, including those inside a scalar subquery or a
    later CTE. A CTE's own `WITH name AS (…)` header is an alias, not an `exp.Table`, so it is
    not counted — only real references are.
    """
    from sqlglot import expressions as exp

    return sum(1 for t in root.find_all(exp.Table) if t.name == name)


def _align_setop_columns(left: Dataset, right: Dataset) -> Dataset:
    """Rename `right`'s columns to `left`'s positionally, for a set operation.

    SQL set ops (UNION/INTERSECT/EXCEPT) pair columns by position and adopt the
    left query's names. The engine's `union`/`intersect`/`except_` require matching
    names, so re-map the right side before handing it over.
    """
    lc, rc = left.columns, right.columns
    if len(lc) != len(rc):
        raise PlanError(
            f"set operation needs both sides to have the same number of columns: "
            f"left has {len(lc)} {lc}, right has {len(rc)} {rc}"
        )
    mapping = {old: new for old, new in zip(rc, lc, strict=True) if old != new}
    return right.rename(mapping) if mapping else right


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

    def _cte_dataset(self, root, cte) -> Dataset:
        """The `Dataset` a CTE binds to — *materialized* when it is referenced more than once.

        A CTE is otherwise a lazy plan, so every `FROM cte` inlines it and **re-executes** the
        whole subtree. TPC-H q15 references its `revenue` CTE twice — once in the join, once in
        `(SELECT max(total_revenue) FROM revenue)` — and so scanned, filtered and grouped 6M
        lineitem rows twice: **46.9 ms, against 7.6 ms** computing it once. This is what DuckDB
        does with a multiply-referenced CTE.

        It also forecloses a real hazard rather than one we hit: re-executing a *float* aggregate
        can legitimately differ in the last ULP between evaluations (different scheduling ⇒
        different summation order, and float addition is not associative), and q15 compares two
        evaluations of that sum for **equality**. Batcher happened to agree with DuckDB either
        way; Daft, which inlines, returns **0 rows instead of 1 on 3 runs in 4** on this exact
        query. One evaluation makes both references read identical bytes, so the question cannot
        arise.

        Referenced once ⇒ left lazy, so predicate/projection pushdown still reaches into it.
        """
        ds = self.statement(cte.this)
        if _table_ref_count(root, cte.alias) > 1:
            return from_arrow(ds.collect())
        return ds

    def _recursive_cte(self, cte) -> Dataset:
        """Evaluate a `WITH RECURSIVE` CTE to a fixpoint, eagerly.

        A recursive CTE is `anchor UNION [ALL] recursive-term`, where the recursive term
        references the CTE itself. It is evaluated the way the SQL standard defines it: run
        the anchor, then repeatedly run the recursive term against *only the rows the last
        iteration produced*, accumulating results, until an iteration yields nothing.

        This is necessarily **eager** — the fixpoint has to run before anything downstream
        can read the CTE — so it materializes, unlike an ordinary lazy CTE. That is also why
        it is bounded: a non-terminating recursion (a missing or wrong stop predicate) would
        otherwise hang, so it raises past `_MAX_RECURSION` iterations rather than spin.

        Args:
            cte: The `CTE` node whose body is self-referential.

        Returns:
            A materialized `Dataset` over the accumulated rows.
        """
        import pyarrow as pa
        from sqlglot import expressions as exp

        body = cte.this
        if not isinstance(body, exp.Union):
            raise NotImplementedError(
                "a recursive CTE must be `<anchor> UNION [ALL] <recursive term>`; "
                f"got {type(body).__name__.lower()}"
            )
        anchor_node, step_node = body.this, body.expression
        if _references(anchor_node, cte.alias):
            raise NotImplementedError(
                "the first branch of a recursive CTE is the anchor and must not reference "
                f"{cte.alias!r}; put the recursive branch second"
            )
        distinct = bool(body.args.get("distinct"))

        frontier = self.statement(anchor_node).collect()
        # Column names come from the CTE header (`c(n)`) when given, else the anchor's.
        names = [c.name for c in (cte.args.get("alias").columns or [])] or frontier.column_names
        frontier = frontier.rename_columns(names)
        accumulated = [frontier]

        saved = self._registry.get(cte.alias)
        try:
            for _ in range(_MAX_RECURSION):
                if frontier.num_rows == 0:
                    break
                # The recursive term sees only the previous iteration's rows — that is what
                # makes this terminate for the usual `SELECT n+1 FROM c WHERE n < k` shape.
                self._registry[cte.alias] = from_arrow(frontier)
                produced = self.statement(step_node).collect().rename_columns(names)
                if distinct and produced.num_rows:
                    # `UNION` (not ALL) is a set fixpoint: rows already derived must not be
                    # fed forward, or a step that keeps re-deriving them never terminates
                    # (`SELECT 1 FROM c` is the degenerate case). Anti-join through the
                    # engine rather than comparing rows in Python.
                    seen = pa.concat_tables(accumulated, promote_options="default")
                    produced = (
                        from_arrow(produced).join(from_arrow(seen), on=names, how="anti").collect()
                    )
                frontier = produced
                accumulated.append(frontier)
            else:
                raise NotImplementedError(
                    f"recursive CTE {cte.alias!r} did not terminate within "
                    f"{_MAX_RECURSION} iterations; check its stop condition"
                )
        finally:
            if saved is None:
                self._registry.pop(cte.alias, None)
            else:
                self._registry[cte.alias] = saved

        # Keep an empty table when nothing was derived, so the CTE still carries the
        # anchor's schema rather than collapsing to "no columns".
        non_empty = [t for t in accumulated if t.num_rows]
        out = (
            pa.concat_tables(non_empty, promote_options="default") if non_empty else accumulated[0]
        )
        ds = from_arrow(out)
        return ds.distinct() if distinct else ds

    # --- statement ---------------------------------------------------------
    def statement(self, node) -> Dataset:
        """Translate a top-level statement: a SELECT or a set operation."""
        from sqlglot import expressions as exp

        # WITH name AS (...), ... — translate each CTE in order and register it
        # under its alias so later FROM references resolve. CTEs may reference
        # earlier ones (they are translated and registered sequentially).
        with_ = node.args.get("with") or node.args.get("with_")
        if with_ is not None:
            recursive = bool(with_.args.get("recursive"))
            for cte in with_.expressions:
                if recursive and _is_self_referential(cte):
                    self._registry[cte.alias] = self._recursive_cte(cte)
                else:
                    self._registry[cte.alias] = self._cte_dataset(node, cte)
            # Strip the WITH so the body translates as an ordinary statement.
            node = node.copy()
            node.set("with", None)
            node.set("with_", None)

        # sqlglot sets distinct=True for the bare set operator, False for its ALL form.
        # Honor it on all three: dropping it on INTERSECT/EXCEPT would silently answer
        # an `ALL` query with DISTINCT multiplicity.
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            left = self.statement(node.this)
            right = self.statement(node.expression)
            # SQL set operations combine by column *position*, taking the left query's
            # output names — not by name. Align the right side's names to the left's,
            # or an operand whose columns merely differ in name (`... id ... UNION
            # ... dept_id ...`) is wrongly rejected as "identical columns" required.
            right = _align_setop_columns(left, right)
            distinct = bool(node.args.get("distinct"))
            if isinstance(node, exp.Union):
                ds = left.union(right, distinct=distinct)
            elif isinstance(node, exp.Intersect):
                ds = left.intersect(right, distinct=distinct)
            else:
                ds = left.except_(right, distinct=distinct)
            # ORDER BY / LIMIT / OFFSET can trail a set operation and apply to its
            # combined result. Ignoring them silently returned unordered / unlimited
            # rows (e.g. `... UNION ALL ... LIMIT 3` kept every row).
            return self._apply_setop_tail(node, ds)
        if isinstance(node, exp.Select):
            return self.select(node)
        if isinstance(node, exp.Values):
            # A bare `VALUES (..), (..)` statement is an inline literal relation.
            return from_clause._values_table(node)
        if isinstance(node, exp.Command) and str(node.this).upper() == "EXPLAIN":
            # sqlglot does not model EXPLAIN; it parses as a Command carrying the rest
            # of the query as text. Re-parse it, render the *planned* tree (no
            # execution), and hand it back as a one-row relation like DuckDB's EXPLAIN.
            return self._explain(node)
        # A semicolon-separated script parses as one Block. Saying "got Block" tells a
        # user nothing about what they typed, so name the actual cause.
        if type(node).__name__ == "Block":
            raise PlanError(
                "bt.sql() runs one statement; this query has several separated by ';'. "
                "Call it once per statement, keeping each result as a Dataset."
            )
        raise PlanError(
            f"cannot translate a {type(node).__name__} statement into a relation; the "
            "query translator handles SELECT / UNION / INTERSECT / EXCEPT / VALUES and "
            "EXPLAIN. (CREATE/DROP and the DML statements are dispatched before this "
            "point, so reaching here means the statement form is not supported at all.)"
        )

    def _explain(self, node) -> Dataset:
        """Translate an ``EXPLAIN [ANALYZE] <query>`` command into a plan relation."""
        import sqlglot

        text = node.args["expression"].this if node.args.get("expression") else ""
        analyze = False
        stripped = text.lstrip()
        if stripped[:8].upper() == "ANALYZE ":
            analyze, text = True, stripped[8:]
        inner = sqlglot.parse_one(text, read="duckdb")
        plan = self.statement(inner).explain(analyze=analyze)
        return _as_dataset(pa.table({"explain_key": ["plan"], "explain_value": [plan]}))

    def _apply_setop_tail(self, node, ds: Dataset) -> Dataset:
        """Apply a trailing ORDER BY / LIMIT / OFFSET on a set-operation result."""
        import sys

        from sqlglot import expressions as exp

        order = node.args.get("order")
        if order is not None:
            # Positional ORDER BY (`ORDER BY 1`) resolves against the leftmost
            # SELECT's projection list (the set op's output columns).
            leftmost = node.this
            while not isinstance(leftmost, exp.Select):
                leftmost = leftmost.this
            ds = self._order(ds, order, leftmost.expressions)
        limit = node.args.get("limit")
        offset = node.args.get("offset")
        if limit is not None or offset is not None:
            skip = int(offset.expression.this) if offset is not None else 0
            n = int(limit.expression.this) if limit is not None else sys.maxsize
            ds = ds.limit(n, offset=skip)
        return ds

    # --- clause building (clauses.py / from_clause.py) ---------------------
    def select(self, node) -> Dataset:
        return clauses._select(self, node)

    def _from(self, node) -> Dataset:
        return from_clause._from(self, node)

    def _table(self, node) -> Dataset:
        return from_clause._table(self, node)

    def _order(self, ds: Dataset, order, projections=None) -> Dataset:
        return clauses._order(self, ds, order, projections)

    # --- registered Python functions (udf.py) ------------------------------
    def _hoist_udfs(self, ds: Dataset, clause_nodes):
        from batcher._sql.parser import udf

        return udf._hoist_udfs(self, ds, clause_nodes)

    # --- subquery decorrelation (subquery.py) ------------------------------
    def _apply_subquery_predicates(self, ds: Dataset, pred):
        return subquery._apply_subquery_predicates(self, ds, pred)

    def _decorrelate_scalar_subqueries(self, ds: Dataset, roots, outer_node=None) -> Dataset:
        return subquery._decorrelate_scalar_subqueries(self, ds, roots, outer_node)

    def _reject_correlated(self, select_node) -> None:
        subquery._reject_correlated(select_node)

    def _disambiguate_columns(self, select_node) -> None:
        _disambiguate_columns(self, select_node)

    # --- grouping / aggregation (grouping.py) ------------------------------
    def _grouping_sets_union(self, node, group) -> Dataset:
        return grouping_sets._grouping_sets_union(self, node, group)

    def _projection_map(self, ds: Dataset, projections) -> dict[str, Expr]:
        return grouping._projection_map(self, ds, projections)

    def _aggregate(
        self, ds: Dataset, projections, group, having, windows=None, order=None
    ) -> tuple:
        return grouping._aggregate(self, ds, projections, group, having, windows, order)

    def _distinct_on(self, ds: Dataset, projections, order, on_exprs) -> Dataset:
        return grouping._distinct_on(self, ds, projections, order, on_exprs)

    # --- window functions (windowing.py) -----------------------------------
    def _is_window(self, p) -> bool:
        return windowing._is_window(p)

    def _inline_named_windows(self, node) -> None:
        windowing._inline_named_windows(node)

    def _window(self, ds: Dataset, projections) -> Dataset:
        return windowing._window(ds, projections)

    # --- scalar expressions (expressions/) ---------------------------------
    def _scalar(self, node) -> Expr:
        return expressions._scalar(self, node)

    # --- shared AST helpers (core_utils.py) --------------------------------
    def _has_aggregate(self, node) -> bool:
        return _has_aggregate(node)

    def _alias_of(self, p) -> str:
        return _alias_of(p)
