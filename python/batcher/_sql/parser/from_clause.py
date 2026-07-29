"""FROM / JOIN / UNNEST / VALUES translation for the SQL translator.

The relation-producing half of a SELECT: what the query reads before any projection,
grouping or ordering applies. Split from `clauses.py` (which keeps the SELECT-shape
logic) because it is a distinct responsibility with its own vocabulary — table
resolution, join strategy, lateral unnest, inline literal relations — and because the
two together outgrew the module size limit.

Functions take the translator instance (`tr`) as their first argument.
"""

from __future__ import annotations

import pyarrow as pa
from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser import udf
from batcher._sql.parser.joins import and_conjuncts as _and_conjuncts
from batcher._sql.parser.joins import outer_theta_join, swap_on_sides
from batcher._sql.parser.joins.lateral import lateral_select, lateral_unnest
from batcher.api.dataset import Dataset
from batcher.api.session import from_arrow
from batcher.plan.expr_ir import col, lit
from batcher.plan.schema import suggest_columns


def _from(tr, node) -> Dataset:
    from_ = node.args.get("from_") or node.args.get("from")
    if from_ is None:
        # `SELECT <exprs>` with no FROM → one row of constants (e.g.
        # `SELECT 1 + 1`, `SELECT extract(year from date '2021-01-01')`).
        return from_arrow(pa.table({"__dummy": [0]}))
    ds = _table(tr, from_.this)

    for join in node.args.get("joins", []) or []:
        # `FROM t, UNNEST(t.xs) AS u(x)` / `FROM t CROSS JOIN UNNEST(...)` — a lateral
        # unnest, which is `Dataset.explode` rather than a join: the elements come from
        # the left row, so there is no right relation to join against.
        if isinstance(join.this, exp.Unnest):
            ds = lateral_unnest(ds, join)
            continue
        # `FROM t, LATERAL (SELECT <exprs>)` — a lateral with no FROM of its own computes
        # per-row values from the outer row, which is exactly `with_columns`.
        if isinstance(join.this, exp.Lateral):
            ds = lateral_select(tr, ds, join.this)
            continue
        right = _table(tr, join.this)
        on = join.args.get("on")
        using = join.args.get("using")
        natural = (join.args.get("method") or "").upper() == "NATURAL"
        how = _join_how(join)
        if how.endswith("_swapped"):
            # RIGHT SEMI/ANTI: run it as a left-driven semi/anti over swapped operands.
            # The ON predicate must be swapped too — `_split_join_on` reads the equality's
            # operand *position* to decide which side a key belongs to, so leaving
            # `ON a.x = b.y` untouched would bind `x` to the new left relation (b).
            how = how.removesuffix("_swapped")
            ds, right = right, ds
            on = swap_on_sides(on) if on is not None else None
        if natural:
            # NATURAL JOIN is USING every column the two sides share, in left order.
            # (Without it we would fall through to the cross-join branch and silently
            # return a cartesian product.)
            keys = [c for c in ds.columns if c in set(right.columns)]
            if not keys:
                raise PlanError(
                    "NATURAL JOIN needs at least one shared column name between the "
                    f"two relations; left has {ds.columns}, right has {right.columns}"
                )
            ds = ds.join(right, on=keys, how=how)
        elif using:
            keys = [u.name for u in using]
            ds = ds.join(right, on=keys, how=how)
        elif on is None:
            # No ON/USING → cross join (cartesian product), expressed as an
            # inner join on a constant key that is then dropped.
            ck = "__cross_key"
            ds = (
                ds.with_columns(**{ck: lit(1)})
                .join(right.with_columns(**{ck: lit(1)}), on=ck)
                .drop(ck)
            )
        else:
            ds = _join_on(tr, ds, right, on, how)
    return ds


def _join_how(join) -> str:
    """The join type for one `JOIN` clause, combining sqlglot's `side` and `kind`.

    `side` carries LEFT/RIGHT/FULL; `kind` separately carries SEMI/ANTI (a bare
    ``SEMI JOIN`` is ``side='' kind='SEMI'``, and ``LEFT SEMI JOIN`` is
    ``side='LEFT' kind='SEMI'``). Reading only `side` — as this did — dropped the kind
    and silently ran every semi/anti join as an ordinary INNER or RIGHT join, so
    ``ANTI JOIN`` returned the rows that *matched*, with the right side's columns
    attached. A wrong answer, not an error, which is why both halves are read here.

    Args:
        join: The sqlglot `Join` node.

    Returns:
        The `Dataset.join` ``how`` — one of inner/left/right/full/semi/anti.
    """
    side = (join.side or "inner").lower()  # "" → inner; "LEFT" → left; "FULL" → full
    how = side if side in {"inner", "left", "right", "full"} else "inner"
    kind = (join.kind or "").upper()
    if kind not in {"SEMI", "ANTI"}:
        return how
    if how == "right":
        # `A RIGHT SEMI JOIN B` is exactly `B SEMI JOIN A`: it returns the rows of the
        # RIGHT relation that have (or lack) a match. Swapping the operands is therefore
        # the whole rewrite — and it is the *correct* one, because a right-semi's output
        # is the right side's columns, which is precisely what the swap yields. Signalled
        # to the caller as a "<kind>_swapped" how; adding a RightSemi/RightAnti variant to
        # the engine's JoinType instead would fork the shared wire-contract enum across
        # four crates to express something the operand order already says.
        return f"{kind.lower()}_swapped"
    return kind.lower()


def _join_on(tr, ds: Dataset, right: Dataset, on, how: str) -> Dataset:
    """Join on an ``ON`` predicate: equi-keys drive the hash join, the rest post-filters.

    A pure equi-join (``a=b`` or ``a=b AND c=d``) joins directly. A mixed predicate
    (``a=b AND x<y``) keeps the equality conjuncts as join keys and applies the
    remaining conjuncts as a filter on the joined result.

    A predicate with **no** equality conjunct is a pure theta join (``ON a.x < b.y``).
    For an INNER join that is exactly ``cross join + filter`` — the definition of a
    nested-loop join — so it is lowered to that rather than rejected. It is O(left x
    right) by nature, and the cross product materializes before the filter, so it is
    for small-to-moderate inputs; an equality conjunct is dramatically better whenever
    the predicate admits one.

    An OUTER pure theta join is still rejected: cross+filter drops the unmatched rows
    an outer join must null-extend, and preserving them needs a real nested-loop join
    operator in the engine (tracked in `docs/internals/databricks_parity.md`).
    """
    eq_pairs, extra = _split_join_on(on)
    if not eq_pairs:
        _reject_ambiguous_residual(on, ds, right, set())
        if how == "inner":
            return ds.cross_join(right).filter(tr._scalar(on))
        if how in {"left", "right", "full"}:
            return outer_theta_join(tr, ds, right, on, how)
        raise NotImplementedError(
            f"{how} join with a pure non-equi/theta ON condition is not supported "
            "(inner, left, right and full are); add an equality conjunct (ON a=b AND ...)"
        )
    left_keys = [lk for lk, _ in eq_pairs]
    right_keys = [rk for _, rk in eq_pairs]
    # An ON residual on an outer join can't be a post-join filter — that would drop
    # the null-extended rows. Pre-filter the nullable side instead (or reject).
    if extra is not None and how != "inner":
        ds, right, extra = _outer_join_residual(tr, ds, right, extra, how)
    if extra is not None:
        _reject_ambiguous_residual(extra, ds, right, set(left_keys) | set(right_keys))
    if left_keys == right_keys:
        # Indistinguishable to the alias-blind resolver, so the merged column is the only
        # answer available. `_disambiguate_columns` renames an `ON`-form same-name key pair
        # apart before we get here, so this is the residue it could not reach (a subquery
        # or CTE side whose columns it cannot enumerate).
        ds = ds.join(right, on=left_keys, how=how)
    elif how in {"semi", "anti"}:
        # A semi/anti join emits the left side's columns only — nothing is coalesced away.
        ds = ds.join(right, left_on=left_keys, right_on=right_keys, how=how)
    else:
        ds = _join_keeping_both_keys(ds, right, left_keys, right_keys, how)
    if extra is not None:
        ds = ds.filter(tr._scalar(extra))
    return ds


def _join_keeping_both_keys(
    left: Dataset, right: Dataset, left_keys: list[str], right_keys: list[str], how: str
) -> Dataset:
    """Join on an ``ON`` equality without merging the two sides' key columns.

    `Dataset.join` *coalesces* a key pair: the output carries one column, under the left
    key's name, holding whichever side matched. That is right for the DataFrame API and
    for SQL's ``USING`` / ``NATURAL`` forms, which do specify a single merged key. It is
    wrong for SQL's ``ON`` form, where the two keys stay separate columns and each is
    NULL-extended on its own side — ``L RIGHT JOIN R ON L.k = R.k`` must report ``L.k`` as
    NULL for a right row that matched nothing. Reusing the coalesced column for both made
    ``L.k`` echo the right side's value: a silent wrong answer, invisible to an
    order-independent differential check that never selected both keys.

    So each side's key is copied to a shadow column that the join cannot coalesce, and the
    real key names are restored from the shadows afterwards. The shadow of an unmatched
    row is NULL-extended by the join exactly as any other payload column would be, which
    is precisely the SQL semantics.

    Args:
        left: The left relation.
        right: The right relation.
        left_keys: The left side's equi-join key columns.
        right_keys: The right side's equi-join key columns, positionally paired.
        how: The join type — inner/left/right/full (semi/anti never reach here).

    Returns:
        The joined dataset, carrying both sides' key columns under their own names.
    """
    shadow_l = {f"__jk_l{i}": k for i, k in enumerate(left_keys)}
    shadow_r = {f"__jk_r{i}": k for i, k in enumerate(right_keys)}
    joined = left.with_columns(**{s: col(k) for s, k in shadow_l.items()}).join(
        right.with_columns(**{s: col(k) for s, k in shadow_r.items()}),
        left_on=left_keys,
        right_on=right_keys,
        how=how,
    )
    # Left keys first, so a right key that happens to share a left key's name still wins
    # its own column — the reference that named it came from the right side.
    restore = {k: col(s) for s, k in shadow_l.items()}
    restore.update({k: col(s) for s, k in shadow_r.items()})
    return joined.with_columns(**restore).drop(*shadow_l, *shadow_r)


def _outer_join_residual(tr, left: Dataset, right: Dataset, extra, how: str):
    """Resolve a non-equi ON residual on an outer join by pre-filtering the nullable side.

    In ``A LEFT JOIN B ON A.k = B.k AND <residual>``, the residual filters which B
    rows are eligible to match — it is *not* a predicate on the result (B columns are
    null where nothing matched, and those left rows must survive). When the residual
    references only the null-extended side, applying it to that side before the join
    is exactly correct. A residual touching the preserved side, or a FULL join (both
    sides preserved), cannot be expressed this way and is rejected rather than
    silently mis-answered. Returns ``(left, right, remaining_residual_or_None)``.
    """
    refs = {c.name for c in extra.find_all(exp.Column)}
    left_cols, right_cols = set(left.columns), set(right.columns)
    if how == "left" and refs <= right_cols and not (refs & left_cols):
        return left, right.filter(tr._scalar(extra)), None
    if how == "right" and refs <= left_cols and not (refs & right_cols):
        return left.filter(tr._scalar(extra)), right, None
    raise NotImplementedError(
        f"{how} join with a non-equi ON condition that references the preserved side "
        f"(or a FULL join) is not supported; the engine join is equi-only — move the "
        f"condition to a WHERE clause or pre-filter the table"
    )


def _reject_ambiguous_residual(extra, left: Dataset, right: Dataset, keys: set[str]) -> None:
    """Reject a residual join condition that references a name present on both sides.

    The residual is applied as a post-join filter, where table qualifiers are lost
    (``a.v`` and ``b.v`` both resolve to ``v``), so a collision would be evaluated
    against the wrong column. Surface it instead of returning a wrong answer.
    """
    collisions = (set(left.columns) & set(right.columns)) - keys
    referenced = {c.name for c in extra.find_all(exp.Column)}
    ambiguous = sorted(referenced & collisions)
    if ambiguous:
        raise NotImplementedError(
            f"join condition references column(s) {ambiguous} present on both sides; "
            f"rename/alias them or apply the non-equi condition as a post-join filter"
        )


def _split_join_on(on):
    """Split an ``ON`` predicate into ``(equi key pairs, residual predicate)``."""
    eq_pairs: list[tuple[str, str]] = []
    residual: list = []
    for conj in _and_conjuncts(on):
        if (
            isinstance(conj, exp.EQ)
            and isinstance(conj.this, exp.Column)
            and isinstance(conj.expression, exp.Column)
        ):
            eq_pairs.append((conj.this.name, conj.expression.name))
        else:
            residual.append(conj)
    extra = None
    for term in residual:
        extra = term if extra is None else exp.And(this=extra, expression=term)
    return eq_pairs, extra


def _table(tr, node) -> Dataset:
    # A PIVOT / UNPIVOT modifier reshapes the table. sqlglot attaches it as `pivots`;
    # it maps onto the relational `Dataset.pivot` / `unpivot` the engine already has, so
    # it is applied rather than rejected. Deferred until after the base relation is
    # resolved (it needs the table's column list to infer the index columns).
    pivots = node.args.get("pivots") if getattr(node, "args", None) else None

    # FROM f(t) — a registered table function (`f` wraps the relation argument).
    if isinstance(node, exp.Table) and isinstance(node.this, exp.Anonymous):
        fname = node.this.name
        rf = tr._functions.get(fname)
        if rf is None:
            raise PlanError(f"unknown table function {fname!r}; registered: {list(tr._functions)}")
        if not rf.table:
            raise PlanError(f"{fname!r} is a scalar function; call it in SELECT, not FROM")
        return _apply_tablesample(udf._apply_table_function(tr, node.this, rf), node)

    # FROM (VALUES (..), (..)) AS t(c1, c2) — an inline literal relation.
    if isinstance(node, exp.Values):
        return _apply_tablesample(_values_table(node), node)

    # FROM (SELECT ...) AS t  → translate the inner SELECT to a Dataset.
    if isinstance(node, exp.Subquery):
        ds = tr.statement(node.this)
    elif isinstance(node, (exp.Select, exp.Union)):
        ds = tr.statement(node)
    else:
        name = node.name
        if name not in tr._registry:
            known = list(tr._registry)
            raise PlanError(
                f"unknown table {name!r}; registered: {known}{suggest_columns(name, known)}"
            )
        ds = tr._registry[name]
    ds = _apply_tablesample(ds, node)
    return _apply_pivots(ds, pivots) if pivots else ds


def _apply_pivots(ds: Dataset, pivots) -> Dataset:
    """Apply a SQL ``PIVOT`` / ``UNPIVOT`` modifier onto the relational Dataset methods.

    `PIVOT (agg(v) FOR k IN ('a','b'))` widens: one output column per listed `k` value,
    each holding `agg(v)` for the rows sharing the remaining columns. `UNPIVOT (val FOR
    name IN (a, b))` is the inverse. Both are exactly `Dataset.pivot` / `Dataset.unpivot`,
    so the modifier is translated rather than rejected — the index columns are whatever the
    relation has left once the pivot's own columns are accounted for.

    Args:
        ds: The base relation the modifier applies to.
        pivots: sqlglot's `pivots` list; exactly one is supported.

    Returns:
        The reshaped dataset.
    """
    if len(pivots) != 1:
        raise NotImplementedError("only a single PIVOT / UNPIVOT modifier is supported")
    piv = pivots[0]
    fields = piv.args.get("fields") or []
    if len(fields) != 1 or not isinstance(fields[0], exp.In):
        raise NotImplementedError("PIVOT / UNPIVOT needs exactly one `<col> IN (...)` clause")
    field = fields[0]
    exprs = piv.args.get("expressions") or []

    if piv.args.get("unpivot"):
        if len(exprs) != 1:
            raise NotImplementedError("UNPIVOT supports a single value column")
        on = [c.name for c in field.expressions]
        index = [c for c in ds.columns if c not in set(on)]
        return ds.unpivot(
            index=index, on=on, variable_name=field.this.name, value_name=exprs[0].name
        )

    if len(exprs) != 1 or not isinstance(exprs[0], exp.AggFunc):
        raise NotImplementedError("PIVOT supports a single aggregate, e.g. `sum(v)`")
    agg = exprs[0]
    if not isinstance(agg.this, exp.Column):
        raise NotImplementedError("PIVOT's aggregate takes a single plain column")
    values = agg.this.name
    on = field.this.name
    # Every listed value becomes an output column; the rest of the relation is the index.
    columns = [str(v.this) if hasattr(v, "this") else str(v) for v in field.expressions]
    index = [c for c in ds.columns if c not in {on, values}]
    return ds.pivot(
        index=index,
        on=on,
        values=values,
        aggregate=type(agg).__name__.lower(),
        columns=columns,
    )


def _values_literal(node):
    """The Python value a VALUES cell denotes (constant literals only)."""
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Neg):
        inner = _values_literal(node.this)
        return None if inner is None else -inner
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        text = node.this
        return float(text) if ("." in text or "e" in text.lower()) else int(text)
    raise NotImplementedError(
        f"VALUES supports only constant literals per cell, got {type(node).__name__}"
    )


def _values_table(node) -> Dataset:
    """Build a `Dataset` from an inline ``VALUES (..), (..)`` relation.

    Column names come from the table alias (``AS t(c1, c2)``) or default to
    ``col0, col1, ...`` (DuckDB's convention). Cells are constant literals; a
    column's type is inferred from its values (mixed NULLs allowed).
    """
    rows = [[_values_literal(cell) for cell in tup.expressions] for tup in node.expressions]
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise PlanError("every VALUES row must have the same number of columns")
    alias = node.args.get("alias")
    named = [c.name for c in alias.args.get("columns", [])] if alias is not None else []
    if named and len(named) != width:
        raise PlanError(
            f"VALUES column-alias count ({len(named)}) does not match the row width ({width})"
        )
    names = named or [f"col{i}" for i in range(width)]
    columns = {names[i]: pa.array([r[i] for r in rows]) for i in range(width)}
    return from_arrow(pa.table(columns))


def _apply_tablesample(ds: Dataset, node) -> Dataset:
    """Apply a SQL ``TABLESAMPLE`` on a table/subquery: ``BERNOULLI(p PERCENT)`` →
    fraction sample, ``RESERVOIR(n ROWS)`` → fixed-count sample. Both lower to
    `Dataset.sample` (deterministic, partition-independent)."""
    sample = node.args.get("sample") if hasattr(node, "args") else None
    if sample is None:
        return ds

    def _num(x):
        return x.this if x is not None and hasattr(x, "this") else x

    percent = _num(sample.args.get("percent"))
    size = _num(sample.args.get("size"))
    if percent is not None:
        return ds.sample(float(percent) / 100.0)
    if size is not None:
        return ds.sample(n=int(size))
    return ds
