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
from sqlglot import expressions as exp

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
from batcher._sql.parser.core_utils import (
    _alias_of,
    _disambiguate_columns,
    _has_aggregate,
    _row_window,
    _unwrap_alias,
)
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow
from batcher.plan.expr_ir import AggExpr, Expr, Lit, col, nullif
from batcher.plan.types import dtype_name

__all__ = ["sql", "translate_ast"]


def sql(
    query: str,
    *,
    dialect: str = "duckdb",
    functions: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    engines: dict[str, Any] | None = None,
    **tables: Dataset | pa.Table,
) -> Dataset:
    """Parse `query` in `dialect` and translate it against the named tables/functions/models.

    A parse failure is re-raised as `PlanError` with a plain-text message; see
    `batcher._internal.sql_errors.parse_sql`.
    """
    return translate_ast(
        parse_sql(query, dialect=dialect),
        functions=functions,
        models=models,
        engines=engines,
        **tables,
    )


def translate_ast(
    ast: Any,
    *,
    functions: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    engines: dict[str, Any] | None = None,
    **tables: Dataset | pa.Table,
) -> Dataset:
    """Translate an already-parsed sqlglot statement into a lazy `Dataset`.

    The string entry point (`sql`) and the session DDL path (which has parsed the
    statement to dispatch ``CREATE``/``DROP``) share this one translator entry.

    `models` is the catalog `ML_PREDICT(t, m)` resolves a model name against; a query that
    scores by path instead needs none. `engines` is its generative counterpart, which
    `AI_GENERATE(t, e)` and friends resolve against; an engine has no path spelling, so a
    query using one always needs it.
    """
    registry = {name: _as_dataset(t) for name, t in tables.items()}
    # Normalize the quantified comparisons into the `IN`/`NOT IN` they are defined as before
    # anything reads the tree, so every clause sees one spelling of set membership rather
    # than each having to learn a second.
    subquery.normalize_quantified(ast)
    return _Translator(registry, functions or {}, models or {}, engines or {}).statement(ast)


# Iteration cap for a recursive CTE. A wrong or missing stop condition would otherwise
# loop forever; failing loudly at a generous bound is better than hanging. DuckDB's
# equivalent guard is 1024 by default.
_MAX_RECURSION = 1024


def _references(node, name: str) -> bool:
    """Whether `node` reads the table `name` anywhere beneath it."""
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
    return sum(1 for t in root.find_all(exp.Table) if t.name == name)


def _align_setop_by_name(left: Dataset, right: Dataset) -> tuple[Dataset, Dataset]:
    """Align two set-operation branches by column *name* (`UNION ... BY NAME`).

    ``BY NAME`` pairs the branches on their column names rather than their positions, and
    a name only one branch has becomes a NULL-filled column on the other. Without it the
    positional rule applied regardless of the modifier, so ``SELECT i, g ... UNION ALL BY
    NAME SELECT g, i ...`` paired an integer with a string and the whole query failed.

    The output column order is the left branch's, then whatever names only the right
    branch has, which is what DuckDB produces.

    Args:
        left: The left branch.
        right: The right branch.

    Returns:
        Both branches, projected onto the same ordered column list.
    """
    order = list(left.columns) + [c for c in right.columns if c not in left.columns]
    left_types = dict(zip(left.columns, left.schema.types, strict=True))
    right_types = dict(zip(right.columns, right.schema.types, strict=True))

    def fill(ds: Dataset, own: dict[str, Any], other: dict[str, Any]) -> Dataset:
        missing = {}
        for name in order:
            if name in own:
                continue
            named = dtype_name(other.get(name))
            typed = nullif(Lit(1), Lit(1))
            missing[name] = typed.cast(named) if named is not None else typed
        if missing:
            ds = ds.with_columns(**missing)
        return ds.select(*order)

    return fill(left, left_types, right_types), fill(right, right_types, left_types)


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


def _bare_null_positions(node) -> set[int]:
    """Output positions a set-operation branch fills with an untyped bare ``NULL``.

    A nested set operation contributes a position only when *every* branch under it is a
    bare NULL there, since one typed branch already fixes the column's type for the rest.
    """
    if isinstance(node, exp.Subquery):
        return _bare_null_positions(node.this)
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        return _bare_null_positions(node.this) & _bare_null_positions(node.expression)
    if not isinstance(node, exp.Select):
        return set()
    return {i for i, p in enumerate(node.expressions) if isinstance(_unwrap_alias(p), exp.Null)}


def _type_untyped_nulls(left: Dataset, right: Dataset, node) -> tuple[Dataset, Dataset]:
    """Give a bare ``NULL`` branch column the type the sibling branch supplies.

    SQL leaves a bare ``NULL`` untyped and lets the set operation decide, so
    ``SELECT s_state ... UNION ALL SELECT NULL AS s_state ...`` is a `Utf8` column in both
    branches. Batcher's IR has no untyped null — a bare ``NULL`` lowers to ``nullif(1, 1)``,
    which is an `Int64` one — so the pair reached the engine as `Utf8` against `Int64`,
    a genuinely irreconcilable mismatch that failed the whole query. TPC-DS q27 and q36
    are that shape, and it is the ordinary spelling of a hand-written rollup.

    The type is read off the sibling branch rather than guessed, and only a *bare* NULL is
    retyped: a column that merely happens to be all-null keeps whatever type it was
    declared with, and a real `Utf8`/`Int64` clash still raises.

    Args:
        left: The left branch, already translated.
        right: The right branch, with its columns aligned to `left`'s names.
        node: The set-operation node, for its branches' select lists.

    Returns:
        The two branches, with any bare-NULL column cast to its sibling's type.
    """
    left_nulls = _bare_null_positions(node.this)
    right_nulls = _bare_null_positions(node.expression)
    # Symmetric difference: a position both sides leave untyped has nothing to adopt, and
    # a position neither leaves untyped needs no help.
    if not (left_nulls ^ right_nulls):
        return left, right
    lt, rt = left.schema.types, right.schema.types
    recast_left: dict[str, Expr] = {}
    recast_right: dict[str, Expr] = {}
    for i, name in enumerate(left.columns):
        if lt[i] == rt[i] or (i in left_nulls) == (i in right_nulls):
            continue
        into, target = (recast_left, rt[i]) if i in left_nulls else (recast_right, lt[i])
        named = dtype_name(target)
        if named is not None:  # a nested/extension type the cast grammar cannot spell
            into[name] = col(name).cast(named)
    if recast_left:
        left = left.with_columns(**recast_left)
    if recast_right:
        right = right.with_columns(**recast_right)
    return left, right


def _as_dataset(t: Dataset | pa.Table) -> Dataset:
    if isinstance(t, Dataset):
        return t
    if isinstance(t, pa.Table):
        return from_arrow(t)
    raise TypeError(f"table must be a Dataset or pyarrow.Table, got {type(t).__name__}")


class _Translator:
    def __init__(
        self,
        registry: dict[str, Dataset],
        functions: dict[str, Any],
        models: dict[str, Any] | None = None,
        engines: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._functions = functions
        # Fitted models `ML_PREDICT` can name. Separate from `_functions` because a model is
        # not callable from an expression: it is scored over a whole relation, in a stage the
        # engine schedules, and giving it its own catalog is what keeps that distinction in
        # the surface rather than in a convention.
        self._models = models or {}
        # Engines `AI_GENERATE`/`AI_CLASSIFY`/`AI_EXTRACT` can name. Separate from `_models`
        # for the same reason that catalog is separate from `_functions`: a language model is
        # reached through a different stage than a fitted estimator, and one catalog holding
        # both would make `ML_PREDICT(t, e)` look legal.
        self._engines = engines or {}
        self._agg_map: dict[str, tuple[str, AggExpr]] | None = None
        # Per select node (by id), which joined-relation column each source contributed:
        # `{alias: {bare column -> column now carrying it}}`. Written by
        # `_disambiguate_columns`, read by a qualified `x.*`.
        self._star_sources: dict[int, dict[str, dict[str, str]]] = {}
        self._agg_n = 0
        # A window's select-list alias may shadow a source column, which the relational
        # window operator cannot express (its output is appended to the input). When that
        # happens the window is materialized under a hidden name; this maps the user's
        # alias to it so the projection and a QUALIFY predicate can read it back.
        self._win_physical: dict[str, str] = {}
        # A window whose input had to be reshaped to reach the operator (an `avg` over a
        # timestamp runs on the microsecond count) needs its output reshaped back; this
        # maps the alias to the function that does it. Read by `_projection_map`.
        self._win_rewrap: dict[str, Any] = {}
        self._win_out_n = 0
        # The column types currently in scope, when the plan can state them statically.
        # SQL has several names whose meaning depends on the *type* of the argument —
        # `epoch_ms(x)` builds a timestamp from an integer but reads one out of a
        # timestamp, `len(x)` counts characters or elements — and the translator used to
        # guess from the AST alone, which is only decidable for a literal. See
        # `column_type`.
        self._scope_types: dict[str, Any] = {}
        self._scope_schema: Any = None
        self._scalar_sub_n = 0
        self._udf_n = 0
        self._win_arg_n = 0

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
            if bool(node.args.get("by_name")):
                left, right = _align_setop_by_name(left, right)
            else:
                right = _align_setop_columns(left, right)
                # A bare `NULL` in one branch has no type of its own; it takes the
                # sibling's.
                left, right = _type_untyped_nulls(left, right, node)
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
        if isinstance(node, exp.Subquery):
            # A parenthesized query is that query, plus whatever ORDER BY / LIMIT the
            # parentheses themselves carry. Writing the branches of a set operation in
            # parentheses is ordinary SQL — TPC-DS q87 chains three of them with EXCEPT —
            # and reaching this point used to mean the whole statement was refused.
            return self._apply_setop_tail(node, self.statement(node.this))
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
            n, skip = _row_window(limit, offset)
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

    def _projection_map(self, ds: Dataset, projections, star_cols=None) -> dict[str, Expr]:
        return grouping._projection_map(self, ds, projections, star_cols)

    def _aggregate(
        self, ds: Dataset, projections, group, having, windows=None, order=None
    ) -> tuple:
        return grouping._aggregate(self, ds, projections, group, having, windows, order)

    def _distinct_on(self, ds: Dataset, projections, order, on_exprs) -> Dataset:
        return grouping._distinct_on(self, ds, projections, order, on_exprs)

    # --- window functions (windowing/) -----------------------------------
    def _is_window(self, p) -> bool:
        return windowing._is_window(p)

    def _inline_named_windows(self, node) -> None:
        windowing._inline_named_windows(node)

    def _window(self, ds: Dataset, projections) -> Dataset:
        # Scoped to this call: an earlier select's window may have recorded a rewrap under
        # the same alias, and applying it here would reshape the wrong column. Cleared
        # before the hoist, which is where a rewrap is recorded.
        self._win_rewrap.clear()
        ds = windowing.hoist_window_args(self, ds, projections)
        return windowing._window(self, ds, projections)

    # --- scalar expressions (expressions/) ---------------------------------
    def _scalar(self, node) -> Expr:
        return expressions._scalar(self, node)

    def bind_scope(self, ds: Dataset) -> None:
        """Record `ds`'s column types, for the name whose meaning depends on them.

        Statically derived (`LogicalPlan.available_schema`), so it costs no rows and is
        simply empty when the plan cannot state a schema — every caller treats it as a
        hint and keeps its old behaviour when a name is absent.
        """
        schema = ds._plan.available_schema()
        self._scope_schema = schema
        self._scope_types = {f.name: f.type for f in schema.arrow} if schema is not None else {}

    def registry_key(self, name: str) -> str | None:
        """The registered table `name` refers to, matched case-insensitively as SQL does.

        Args:
            name: The identifier as written.

        Returns:
            The registry key, or None when nothing matches (or two entries differ only in
            case, which is ambiguous rather than resolvable).
        """
        if name in self._registry:
            return name
        matches = [k for k in self._registry if k.lower() == name.lower()]
        return matches[0] if len(matches) == 1 else None

    def canonicalize_identifiers(self, select_node) -> None:
        """Rewrite this SELECT's column references to the case the relation stores.

        SQL identifiers are case-insensitive, and every engine Batcher is measured against
        treats them so: DuckDB answers ``SELECT I FROM t`` with the column ``i``. The
        relational layer is name-keyed and case-*sensitive*, so an unquoted identifier
        typed in another case failed with "unknown column" — which is most of a ported
        query when the source system upper-cases its DDL.

        Only a name with exactly one case-insensitive match in scope is rewritten, and
        only when the exact spelling is absent, so a relation that genuinely carries both
        ``id`` and ``ID`` is left alone rather than silently resolved to one of them.

        Args:
            select_node: The `Select` whose own column references are canonicalized;
                a nested sub-select owns its columns and is skipped.
        """
        if not self._scope_types:
            return
        folded: dict[str, list[str]] = {}
        for name in self._scope_types:
            folded.setdefault(name.lower(), []).append(name)
        for c in select_node.find_all(exp.Column):
            if c.find_ancestor(exp.Select) is not select_node:
                continue
            name = c.name
            if name in self._scope_types:
                continue
            match = folded.get(name.lower())
            if match is not None and len(match) == 1:
                c.this.set("this", match[0])

    def expr_type(self, expr) -> Any | None:
        """The Arrow type `expr` produces over the columns in scope, or None if unknown.

        The control plane's own static type analysis (`plan.types.infer.infer_type`), which
        answers None rather than guessing — so a caller must treat None as "unknown", never
        as a type.

        Args:
            expr: A built `Expr`.

        Returns:
            The Arrow `DataType`, or None.
        """
        if self._scope_schema is None:
            return None
        from batcher.plan.types.infer import infer_type

        return infer_type(expr, self._scope_schema)

    def column_type(self, node) -> Any | None:
        """The Arrow type of `node` when it is a plain column currently in scope.

        Args:
            node: A sqlglot expression; only a bare `Column` can be resolved.

        Returns:
            The Arrow `DataType`, or None when the node is not a column or the plan
            cannot state its type.
        """
        if not isinstance(node, exp.Column):
            return None
        return self._scope_types.get(node.name)

    # --- shared AST helpers (core_utils.py) --------------------------------
    def _has_aggregate(self, node) -> bool:
        return _has_aggregate(node)

    def _alias_of(self, p) -> str:
        return _alias_of(p)
