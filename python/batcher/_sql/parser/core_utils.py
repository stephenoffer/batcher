"""Small stateless AST helpers shared across translator theme modules.

Kept in their own leaf module so every theme module can import them without
creating an import cycle through the translator class.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import PlanError


def _columns_selector(node) -> Any:
    """Translate DuckDB ``COLUMNS(*)`` / ``COLUMNS('regex')`` to a column selector.

    The selector is expanded against the input schema by the projection builder, so
    ``COLUMNS('^sales')`` projects every matching column and ``func(COLUMNS(*))``
    applies ``func`` to each column. Reuses the DataFrame selector engine.
    """
    from sqlglot import expressions as exp

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


def _positional(projections, literal, clause: str):
    """Resolve a 1-based positional reference (`ORDER BY 2`) to its SELECT item."""
    idx = int(literal.this)
    if not 1 <= idx <= len(projections):
        raise PlanError(
            f"{clause} position {idx} is out of range: the SELECT list has "
            f"{len(projections)} item(s)"
        )
    return projections[idx - 1]


def _unwrap_alias(p):
    from sqlglot import expressions as exp

    return p.this if isinstance(p, exp.Alias) else p


def _alias_of(p) -> str:
    from sqlglot import expressions as exp

    if isinstance(p, exp.Alias):
        return p.alias
    if isinstance(p, exp.Column):
        return p.name
    # No explicit `AS`: derive the output name from the expression, matching the
    # convention of the reference engines (DuckDB/Polars) so a column the user did not
    # alias lines up across engines — `sum(l_quantity)`, `count_star()` — rather than a
    # bespoke `SUM_l_quantity`. `count(*)` is DuckDB's special `count_star()`.
    if isinstance(p, exp.Count) and isinstance(p.this, exp.Star):
        return "count_star()"
    return p.sql().lower()


def _split_and(pred) -> list:
    """Flatten a conjunction (and parentheses) into its leaf predicates."""
    from sqlglot import expressions as exp

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
    from sqlglot import expressions as exp

    out = preds[0]
    for p in preds[1:]:
        out = exp.And(this=out, expression=p)
    return out


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
    from sqlglot import expressions as exp

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
    from sqlglot import expressions as exp

    # An aggregate inside a window (e.g. SUM(x) OVER (...)) is a window
    # function, not a GROUP-BY aggregate, so ignore those. An aggregate
    # inside a (scalar) subquery belongs to the inner query, not this one.
    for a in node.find_all(exp.AggFunc):
        if a.find_ancestor(exp.Window) is not None:
            continue
        if a.find_ancestor(exp.Subquery) is not None:
            continue
        return True
    return False


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
    """
    from sqlglot import expressions as exp

    from_ = node.args.get("from") or node.args.get("from_")
    if from_ is None:
        return
    joins = node.args.get("joins", []) or []
    tables = [t for t in [from_.this, *(j.this for j in joins)] if isinstance(t, exp.Table)]
    if len(tables) < 2:
        return

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
    per_source: list[tuple] = []  # (table_node, alias, columns)
    counts: dict[str, int] = {}
    for t in tables:
        if t.name not in tr._registry:
            if names.count(t.name) > 1:
                raise NotImplementedError(
                    f"self-join on {t.name!r} is not supported (its columns can't be "
                    f"enumerated to disambiguate the aliases)"
                )
            continue
        cols = list(tr._registry[t.name].columns)
        per_source.append((t, t.alias or t.name, cols))
        for c in set(cols):
            counts[c] = counts.get(c, 0) + 1

    shared = {c for c, n in counts.items() if n > 1}
    if natural:  # NATURAL merges every shared column; leave them all bare.
        protected |= shared
    flatten = shared - protected
    shadow_keys = _key_shadows(node, joins, shared & protected)
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
        ).from_(exp.table_(t.name))
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

    Args:
        node: The `Select` node being translated.
        joins: Its `Join` nodes.
        merged_keys: Key names the join merges into one column.

    Returns:
        The subset of `merged_keys` that a qualified reference outside the ON clause names.
    """
    from sqlglot import expressions as exp

    if not merged_keys or any((j.kind or "").upper() in {"SEMI", "ANTI"} for j in joins):
        return set()
    if not any((j.side or "").upper() in {"LEFT", "RIGHT", "FULL"} for j in joins):
        return set()
    return {
        c.name
        for c in node.find_all(exp.Column)
        if c.name in merged_keys
        and c.table
        and c.find_ancestor(exp.Select) is node
        and c.find_ancestor(exp.Join) is None
    }
