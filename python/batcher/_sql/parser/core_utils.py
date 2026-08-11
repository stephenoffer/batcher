"""Small stateless AST helpers shared across translator theme modules.

Kept in their own leaf module so every theme module can import them without
creating an import cycle through the translator class.
"""

from __future__ import annotations

import sys
from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError


def _columns_selector(node) -> Any:
    """Translate DuckDB ``COLUMNS(*)`` / ``COLUMNS('regex')`` to a column selector.

    The selector is expanded against the input schema by the projection builder, so
    ``COLUMNS('^sales')`` projects every matching column and ``func(COLUMNS(*))``
    applies ``func`` to each column. Reuses the DataFrame selector engine.
    """

    from batcher.plan.expr_ir.selectors import all as select_all
    from batcher.plan.expr_ir.selectors import matches

    inner = node.this
    if isinstance(inner, exp.Star):
        return select_all()
    if isinstance(inner, exp.Literal) and inner.is_string:
        return matches(inner.name)
    raise NotImplementedError(
        f"COLUMNS(...) supports COLUMNS(*) or COLUMNS('regex'); got {type(inner).__name__}"
    )


def _row_window(limit, offset) -> tuple[int, int]:
    """The ``(count, skip)`` a ``LIMIT``/``OFFSET``/``FETCH`` clause pair asks for.

    ANSI SQL spells the row cap ``FETCH FIRST n ROWS ONLY``; sqlglot parses that into an
    `exp.Fetch` (holding ``count``) rather than an `exp.Limit` (holding ``expression``),
    and both land in the same ``limit`` slot. Reading only the `Limit` shape crashed the
    ANSI spelling with ``AttributeError: 'NoneType' object has no attribute 'this'`` —
    an internal error for standard SQL — so both node shapes are read here, in one place
    shared by the SELECT and set-operation paths that each used to carry a copy.

    Args:
        limit: The ``limit`` arg — an `exp.Limit`, an `exp.Fetch`, or None.
        offset: The ``offset`` arg — an `exp.Offset` or None.

    Returns:
        The row count to keep (``sys.maxsize`` for "all remaining", which is what a bare
        ``OFFSET`` asks for) and the number of rows to skip.
    """
    skip = int(offset.expression.this) if offset is not None else 0
    if limit is None:
        return sys.maxsize, skip
    # PERCENT takes a fraction of a cardinality nothing has measured yet, and WITH TIES
    # keeps an unbounded number of peers of the last row. Neither is `Dataset.limit`, and
    # both modify a `LIMIT` as well as a `FETCH`. Dropping the modifier and applying the
    # bare row cap is a silent wrong answer: `LIMIT 20 PERCENT` over five rows returned
    # all five where DuckDB returns one. Reject instead.
    options = limit.args.get("limit_options")
    if options is not None:
        for flag, name in (("percent", "PERCENT"), ("with_ties", "WITH TIES")):
            if options.args.get(flag):
                raise NotImplementedError(
                    f"{name} row limits are not supported; use a plain row count "
                    "(LIMIT n / FETCH FIRST n ROWS ONLY)"
                )
    if isinstance(limit, exp.Fetch):
        count = limit.args.get("count")
        # `FETCH FIRST ROW ONLY` omits the count, and sqlglot leaves the bare `ROW`
        # keyword in the slot as an Identifier rather than a number. Standard SQL
        # defaults an omitted count to one.
        if not isinstance(count, exp.Literal):
            return 1, skip
        return int(count.this), skip
    return int(limit.expression.this), skip


def _positional(projections, literal, clause: str):
    """Resolve a 1-based positional reference (`ORDER BY 2`) to its SELECT item."""
    idx = int(literal.this)
    if not 1 <= idx <= len(projections):
        raise PlanError(
            f"{clause} position {idx} is out of range: the SELECT list has "
            f"{len(projections)} item(s)"
        )
    return projections[idx - 1]


def _is_star(p) -> bool:
    """Is this SELECT item a `*` or a qualified `t.*`?"""
    return isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))


def _positional_output(output: list[str], literal, clause: str) -> str:
    """Resolve a 1-based positional reference to the n-th *output* column name.

    The counterpart to :func:`_positional` for a SELECT list carrying a star. A star has no
    single AST item a position can name, and — the part that is easy to miss — it also
    *shifts* every position after it, so counting select-list items is wrong for the whole
    tail of the list rather than only for the star itself. The projected output names are
    the one enumeration that matches what SQL means by "the n-th column".
    """
    idx = int(literal.this)
    if not 1 <= idx <= len(output):
        raise PlanError(
            f"{clause} position {idx} is out of range: the query projects {len(output)} column(s)"
        )
    return output[idx - 1]


def _unwrap_alias(p):
    return p.this if isinstance(p, exp.Alias) else p


def _alias_of(p) -> str:
    if isinstance(p, exp.Alias):
        return p.alias
    # Parentheses around a bare column are grouping, not an expression, so the output name
    # is the column's. `SELECT DISTINCT(x)` parses as `DISTINCT (x)` and is the common way
    # to hit this (TPC-DS q41), which was naming its one output column `(i_product_name)`
    # — enough for the query's own `ORDER BY i_product_name` to fail to resolve.
    bare = p
    while isinstance(bare, exp.Paren):
        bare = bare.this
    if isinstance(bare, exp.Column):
        return bare.name
    # No explicit `AS`: derive the output name from the expression, matching the
    # convention of the reference engines (DuckDB/Polars) so a column the user did not
    # alias lines up across engines — `sum(l_quantity)`, `count_star()` — rather than a
    # bespoke `SUM_l_quantity`. `count(*)` is DuckDB's special `count_star()`.
    if isinstance(p, exp.Count) and isinstance(p.this, exp.Star):
        return "count_star()"
    return p.sql().lower()


def _split_and(pred) -> list:
    """Flatten a conjunction (and parentheses) into its leaf predicates."""
    out: list = []
    stack = [pred]
    while stack:
        p = stack.pop()
        if isinstance(p, exp.And):
            stack.extend((p.this, p.expression))
        elif isinstance(p, exp.Paren):
            stack.append(p.this)
        else:
            out.append(p)
    return out


def _join_and(preds):
    """Re-combine leaf predicates into a single AND chain."""
    out = preds[0]
    for p in preds[1:]:
        out = exp.And(this=out, expression=p)
    return out


def _split_or(pred) -> list:
    """Flatten a disjunction (and parentheses) into its top-level alternatives."""
    out: list = []
    stack = [pred]
    while stack:
        p = stack.pop()
        if isinstance(p, exp.Or):
            stack.extend((p.expression, p.this))
        elif isinstance(p, exp.Paren):
            stack.append(p.this)
        else:
            out.append(p)
    return out


def _factor_common_conjuncts(pred):
    """``(C AND A) OR (C AND B)`` → ``C AND (A OR B)``, for a conjunct shared by every arm.

    Distribution is what hides a correlation from every rewrite in the `subquery` package:
    they all look for the correlating equality among the *top-level conjuncts* of a
    subquery's WHERE, and a query that repeats it inside each arm of an `OR` has no
    top-level conjuncts at all. TPC-DS q41 is exactly that — the same
    ``i_manufact = i1.i_manufact`` in both arms of a two-arm disjunction — and it was
    refused as an unsupported correlated subquery.

    The factoring is a boolean identity, so it is safe whether or not it enables anything,
    and conjuncts are matched by SQL text: an arm repeating a *differently spelled* but
    equivalent predicate simply is not factored, which costs a rewrite rather than
    correctness.

    Args:
        pred: A predicate tree.

    Returns:
        The factored predicate, or `pred` itself when no conjunct is shared by every arm.
    """
    arms = _split_or(pred)
    if len(arms) < 2:
        return pred
    per_arm = [_split_and(a) for a in arms]
    later = [{c.sql() for c in group} for group in per_arm[1:]]
    common = [c for c in per_arm[0] if all(c.sql() in seen for seen in later)]
    if not common:
        return pred
    shared = {c.sql() for c in common}
    remainders = []
    for group in per_arm:
        rest = [c.copy() for c in group if c.sql() not in shared]
        if not rest:
            # This arm is nothing but the shared part, so it subsumes every other arm and
            # the whole disjunction reduces to it.
            return _join_and([c.copy() for c in common])
        remainders.append(_join_and(rest))
    alternatives = remainders[0]
    for r in remainders[1:]:
        alternatives = exp.Or(this=alternatives, expression=r)
    return _join_and([*(c.copy() for c in common), exp.Paren(this=alternatives)])


def _within_group_to_agg(node):
    """`agg(...) WITHIN GROUP (ORDER BY x)` → the ordinary aggregate over `x`.

    Ordered-set aggregates parse as ``WithinGroup(this=<agg>, expression=Order)``
    where the sort column lives in the ``ORDER BY`` and the fraction (if any) sits
    inside the aggregate. Left unrewritten, ``find_all(AggFunc)`` sees the inner
    ``PercentileCont(this=<fraction>)`` and *silently drops the ORDER BY column*,
    treating the fraction as the value column. Rewrite the two forms the engine
    supports into their normal shapes so the aggregate path handles them:

    * ``percentile_cont(f) WITHIN GROUP (ORDER BY x)`` → ``PercentileCont(x, f)``
      (the two-argument quantile form).
    * ``mode() WITHIN GROUP (ORDER BY x)`` → ``Mode(x)``.
    """
    if not isinstance(node, exp.WithinGroup):
        return node
    agg = node.this
    order = node.expression
    ordered = order.expressions if isinstance(order, exp.Order) else []
    if len(ordered) != 1:
        raise NotImplementedError(
            "WITHIN GROUP (ORDER BY ...) requires exactly one ordering expression"
        )
    column = ordered[0].this
    if isinstance(agg, exp.PercentileCont):
        return exp.PercentileCont(this=column.copy(), expression=agg.this.copy())
    if isinstance(agg, exp.Mode):
        return exp.Mode(this=column.copy())
    if isinstance(agg, exp.PercentileDisc):
        raise NotImplementedError(
            "percentile_disc WITHIN GROUP is not supported; use percentile_cont"
        )
    raise NotImplementedError(f"WITHIN GROUP is not supported for {type(agg).__name__.lower()}")


def _has_aggregate(node) -> bool:
    # An aggregate inside a window (e.g. SUM(x) OVER (...)) is a window
    # function, not a GROUP-BY aggregate, so ignore those. An aggregate
    # inside a (scalar) subquery belongs to the inner query, not this one.
    #
    # `iter_agg_nodes`, not `find_all(exp.AggFunc)`: the DuckDB aggregates sqlglot
    # leaves anonymous (`product`, `sem`, `count_star`, …) are not `AggFunc` subclasses,
    # so an ungrouped `SELECT product(x) FROM t` was not recognized as an aggregate
    # query at all and fell through to the scalar translator.

    from batcher._sql.parser.expressions.aggregates import iter_agg_nodes

    for a in iter_agg_nodes(node):
        if a.find_ancestor(exp.Window) is not None:
            continue
        if a.find_ancestor(exp.Subquery) is not None:
            continue
        return True
    return False


_MAX_DERIVED_DEPTH = 8


def _source_columns(tr, node, depth: int = 0) -> list[str] | None:
    """The output column names of one FROM source, or None if they can't be determined.

    Read from the AST rather than by planning the source. Planning it would be exact, but
    `_disambiguate_columns` runs *before* the FROM clause is built and translating a
    subquery twice would advance the translator's alias counters and clobber its
    per-select state — so the names are derived structurally instead.

    Returning None is always safe: the caller then leaves that source alone, which is the
    behavior every derived table had before it was considered at all.

    Args:
        tr: The translator, for its table registry.
        node: A FROM/JOIN source — a base table, a derived table, or a set operation.
        depth: Recursion guard for nested derived tables.

    Returns:
        The source's output column names in order, or None if any part is unresolvable.
    """
    if depth > _MAX_DERIVED_DEPTH:
        return None
    if isinstance(node, exp.Table):
        rel = tr._registry.get(node.name)
        return list(rel.columns) if rel is not None else None
    if isinstance(node, exp.Subquery):
        return _source_columns(tr, node.this, depth + 1)
    if isinstance(node, exp.Union):
        # A set operation takes its column names from its left branch, as SQL specifies.
        return _source_columns(tr, node.this, depth + 1)
    if not isinstance(node, exp.Select):
        return None

    inner_from = node.args.get("from") or node.args.get("from_")
    inner_sources = ([inner_from.this] if inner_from is not None else []) + [
        j.this for j in node.args.get("joins", []) or []
    ]
    out: list[str] = []
    for p in node.expressions:
        # `SELECT *` / `SELECT x.*` — expand from the inner FROM. TPC-DS q44 needs this:
        # its ranked relations are `(SELECT * FROM (SELECT item_sk, ... rnk FROM …) V11 …)`,
        # so the colliding `rnk` is two levels down and invisible to the projection list.
        qualified_star = isinstance(p, exp.Column) and isinstance(p.this, exp.Star)
        if _is_star(p):
            want = p.table if qualified_star else None
            for s in inner_sources:
                if want and (s.alias or getattr(s, "name", None)) != want:
                    continue
                cols = _source_columns(tr, s, depth + 1)
                if cols is None:
                    return None
                out.extend(cols)
            continue
        name = p.alias_or_name
        if not name:
            return None
        out.append(name)
    return out or None


def _disambiguate_columns(tr, node) -> None:
    """Rename colliding columns so the alias-blind resolver sees distinct names.

    Two sources exposing the same column name — two aliases of one table
    (``nation n1, nation n2``), or two different tables sharing a name
    (``emp e, dept d`` both with ``dept``) — otherwise collapse onto one physical
    column, so a qualified ``d.dept`` silently resolves to the *left* ``dept`` (and a
    comma-join ``WHERE e.dept = d.dept`` degenerates to a cartesian product). Each
    source owning a colliding column is wrapped in a subquery renaming it to a flat
    ``alias__col`` name, and matching ``alias.col`` references are rewritten. Columns
    merged by a USING / NATURAL / same-name-``ON``-equi join keep their bare name.

    A merged key keeping its bare name is not the whole story, though: the join coalesces
    the pair, so reusing that one column for a *qualified* ``L.k`` / ``R.k`` made both
    report the coalesced value. Under a RIGHT/FULL join that is a silent wrong answer —
    ``L.k`` echoed the right side's key where SQL requires NULL. So a merged key that the
    query references by qualifier additionally gets a per-side ``alias__col`` *copy*
    (`_key_shadows`), and only the qualified references are redirected onto it. The bare
    name still resolves to the coalesced column, which is what USING/NATURAL specify and
    what the ``ON`` form is leniently allowed here (DuckDB calls that one ambiguous).

    It also records, per select node, which joined-relation column each source contributed
    (``tr._star_sources``). That is what lets ``SELECT x.*`` project x's columns and only
    x's, under their bare names — see :func:`grouping._projection_map`.
    """
    from_ = node.args.get("from") or node.args.get("from_")
    if from_ is None:
        return
    joins = node.args.get("joins", []) or []
    # A *derived* table collides exactly the way a base table does, and is included for
    # that reason: `FROM (SELECT k AS r FROM t) a, (SELECT k AS r FROM t) b WHERE a.r = b.r`
    # otherwise collapsed both `r`s onto one column, so the predicate became `r = r` — true
    # for every pair — and the query silently returned the cartesian product. TPC-DS q44 is
    # that shape (two ranked subqueries joined on `rnk`) and returned 100 rows for 10.
    sources = [
        t
        for t in [from_.this, *(j.this for j in joins)]
        if isinstance(t, (exp.Table, exp.Subquery))
    ]
    if len(sources) < 2:
        return
    tables = [t for t in sources if isinstance(t, exp.Table)]

    # Columns a USING / same-name-ON-equi join merges must keep their bare name (the
    # join unifies them; flattening would make it drop the right key). Comma joins
    # have no ON, so their WHERE equi is not a key here and is still flattened.
    protected: set[str] = set()
    natural = False
    for j in joins:
        protected |= {u.name for u in j.args.get("using") or ()}
        natural = natural or (j.args.get("method") or "").upper() == "NATURAL"
        on = j.args.get("on")
        for eq in on.find_all(exp.EQ) if on is not None else ():
            a, b = eq.this, eq.expression
            if isinstance(a, exp.Column) and isinstance(b, exp.Column) and a.name == b.name:
                protected.add(a.name)

    names = [t.name for t in tables]
    per_source: list[tuple] = []  # (source_node, alias, columns)
    counts: dict[str, int] = {}
    for t in sources:
        cols = _source_columns(tr, t)
        if cols is None:
            # A source whose columns cannot be enumerated cannot be disambiguated. For a
            # base table that is only safe when its name is unique (a self-join would
            # silently collapse); a derived table is always uniquely aliased, so an
            # unresolvable one is simply left alone, exactly as before this change.
            if isinstance(t, exp.Table) and names.count(t.name) > 1:
                raise NotImplementedError(
                    f"self-join on {t.name!r} is not supported (its columns can't be "
                    f"enumerated to disambiguate the aliases)"
                )
            continue
        alias = t.alias or (t.name if isinstance(t, exp.Table) else None)
        if alias is None:
            continue  # an unaliased derived table cannot be referenced by qualifier
        per_source.append((t, alias, cols))
        for c in set(cols):
            counts[c] = counts.get(c, 0) + 1

    shared = {c for c, n in counts.items() if n > 1}
    if natural:  # NATURAL merges every shared column; leave them all bare.
        protected |= shared
    flatten = shared - protected
    shadow_keys = _key_shadows(node, joins, shared & protected)

    # Record where every source's columns end up in the joined relation, so a *qualified*
    # star (`SELECT x.*`) can project exactly that source's columns under their bare names.
    # This has to be recorded even when nothing is renamed below: `x.*` needs to know which
    # columns are x's whether or not any of them collided.
    tr._star_sources[id(node)] = {
        alias: {c: f"{alias}__{c}" if c in flatten or c in shadow_keys else c for c in cols}
        for _, alias, cols in per_source
    }
    if not flatten and not shadow_keys:
        return

    alias_map: dict[str, dict[str, str]] = {}
    shadow_map: dict[str, dict[str, str]] = {}
    for t, alias, cols in per_source:
        flat = {c: f"{alias}__{c}" for c in cols if c in flatten}
        shadow = {c: f"{alias}__{c}" for c in cols if c in shadow_keys}
        if not flat and not shadow:
            continue
        alias_map[alias] = flat
        shadow_map[alias] = shadow
        # Non-colliding columns keep their bare name so an unqualified unique
        # reference (`sal`) still resolves. A shadowed key keeps its bare name *and*
        # gains the `alias__key` copy, so the join still merges the bare pair.
        inner = exp.Select(
            expressions=[
                exp.alias_(exp.column(c), flat[c]) if c in flat else exp.column(c) for c in cols
            ]
            + [exp.alias_(exp.column(c), s) for c, s in shadow.items()]
            # A base table is re-selected by name; a derived table is wrapped around
            # itself, since its rows exist only as its own subquery.
        ).from_(exp.table_(t.name) if isinstance(t, exp.Table) else t.copy())
        t.replace(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias))))

    # A bare `alias.col` projected directly keeps its output name (`col`).
    both = {a: {**alias_map.get(a, {}), **shadow_map.get(a, {})} for a in alias_map}
    for p in list(node.expressions):
        if isinstance(p, exp.Column) and p.name in both.get(p.table, ()):
            p.replace(exp.alias_(exp.column(both[p.table][p.name]), p.name))
    for c in list(node.find_all(exp.Column)):
        if c.find_ancestor(exp.Select) is not node:
            continue
        # Inside the join's own ON, a shadowed key must stay bare: it is what the join
        # keys on, and redirecting it would un-merge the pair the shadow exists beside.
        table = alias_map if c.find_ancestor(exp.Join) is not None else both
        if c.name in table.get(c.table, ()):
            c.replace(exp.column(table[c.table][c.name]))


def _key_shadows(node, joins, merged_keys: set[str]) -> set[str]:
    """The merged join keys that need a per-side copy: those referenced by a qualifier.

    A copy is only worth materializing where the query actually asks a side for its own
    key (``SELECT L.k, R.k``); an unqualified query, or ``SELECT *``, wants the merged
    column and would only be confused by extra output columns. Two whole shapes need no
    copy at all:

    * **No outer join.** An inner join emits only matched rows, so the merged key is
      provably equal to both sides' — there is nothing for a copy to say differently.
    * **Semi/anti.** They emit the left relation's columns alone, so there is no second
      side to tell apart and nothing was coalesced away to begin with.

    A **qualified star** counts as naming every merged key, and has to be handled
    separately: `x.*` asks x for all of its own columns, its share of the key included, but
    the node sqlglot produces for it is a `Column` whose `name` is `"*"` — so matching on
    the key name alone silently misses it. Under a FULL join that showed up as
    `SELECT x.* FROM x FULL JOIN y USING (k)` reporting y's key on a y-only row, where SQL
    requires NULL.

    Args:
        node: The `Select` node being translated.
        joins: Its `Join` nodes.
        merged_keys: Key names the join merges into one column.

    Returns:
        The subset of `merged_keys` that a qualified reference outside the ON clause names.
    """
    if not merged_keys or any((j.kind or "").upper() in {"SEMI", "ANTI"} for j in joins):
        return set()
    if not any((j.side or "").upper() in {"LEFT", "RIGHT", "FULL"} for j in joins):
        return set()
    qualified = {
        c.name
        for c in node.find_all(exp.Column)
        if c.table and c.find_ancestor(exp.Select) is node and c.find_ancestor(exp.Join) is None
    }
    return set(merged_keys) if "*" in qualified else merged_keys & qualified
