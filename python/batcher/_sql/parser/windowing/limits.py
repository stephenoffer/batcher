"""``LIMIT … PERCENT`` and ``FETCH … WITH TIES``, answered with ranking windows.

Layer: `_sql` (surface). Neither modifier is a row cap: ``PERCENT`` takes a fraction of a
cardinality nothing has measured when the plan is built, and ``WITH TIES`` keeps every peer
of the last kept row, however many there are. Both are exact over three windows the engine
already computes, so the statement is rewritten before translation:

    SELECT … ORDER BY k [OFFSET m] LIMIT c [PERCENT] [WITH TIES]
      ->  SELECT * FROM (SELECT …, row_number() OVER (ORDER BY k) AS __bc_lim_rn,
                                   rank()       OVER (ORDER BY k) AS __bc_lim_rk,
                                   count(*)     OVER ()           AS __bc_lim_n
                         FROM …)
          WHERE __bc_lim_rn > m AND <cut>
          ORDER BY __bc_lim_rn

where the row count is ``c``, or ``floor(c * n / 100)`` for ``PERCENT`` (DuckDB's rounding,
and its 0-100 range), and the cut is ``__bc_lim_rn <= m + count`` — or, with ties, a rank no
greater than the rank of the last row that count keeps. The windows share one operator, so
the row number and the rank read one order. The hidden columns carry the ``__bc_`` prefix
that keeps them out of ``SELECT *``.

A ``DISTINCT`` select, or a set operation, is wrapped first and ranked over its output, since
a window added to its select list would change what is distinct; its ORDER BY names output
columns, as SQL requires there.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError

__all__ = ["normalize_limit_modifiers"]

_RN, _RK, _N, _CUT, _SRC = "__bc_lim_rn", "__bc_lim_rk", "__bc_lim_n", "__bc_lim_cut", "__bc_lim"


def normalize_limit_modifiers(ast):
    """Rewrite every ``PERCENT`` / ``WITH TIES`` limit under `ast` into ranking windows.

    Args:
        ast: The parsed statement.

    Returns:
        The statement, rewritten; a new root when the root itself carried the modifier.

    Raises:
        PlanError: For a ``PERCENT`` outside 0-100 or a non-constant count.
        NotImplementedError: For ``WITH TIES`` without ORDER BY, or an ORDER BY key a
            wrapped select cannot resolve to an output column.
    """
    # Deepest first, so rewriting an outer query never copies a node still to be visited.
    targets = [n for n in ast.find_all(exp.Select, exp.Union) if _modifiers(n) is not None]
    for node in reversed(targets):
        rewritten = _rewrite(node)
        if node is ast:
            ast = rewritten
        else:
            node.replace(rewritten)
    return ast


def _modifiers(node) -> tuple[bool, bool] | None:
    """``(percent, with_ties)`` for a node whose limit carries either, else None."""
    limit = node.args.get("limit")
    options = limit.args.get("limit_options") if limit is not None else None
    if options is None:
        return None
    percent, ties = bool(options.args.get("percent")), bool(options.args.get("with_ties"))
    return (percent, ties) if percent or ties else None


def _rewrite(node):
    percent, ties = _modifiers(node)
    limit, offset, order = node.args["limit"], node.args.get("offset"), node.args.get("order")
    count = _constant(limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression)
    if count is None:
        # `FETCH FIRST ROW WITH TIES` omits the count, which standard SQL reads as one.
        count = 1
    if percent and not 0 <= count <= 100:
        raise PlanError(f"LIMIT {count} PERCENT is out of range: it must be between 0 and 100")
    if ties and order is None:
        raise NotImplementedError("WITH TIES needs an ORDER BY to say which rows are tied")
    skip = int(_constant(offset.expression) or 0) if offset is not None else 0

    body = node.copy()
    for key in ("limit", "offset", "order"):
        body.set(key, None)
    keys = _order_keys(node, order) if order is not None else []
    ranked_by = exp.Order(expressions=keys) if keys else None
    windows = [exp.alias_(exp.Window(this=exp.RowNumber(), order=ranked_by), _RN)]
    if ties:
        windows.append(exp.alias_(exp.Window(this=exp.Rank(), order=ranked_by.copy()), _RK))
    if percent:
        windows.append(exp.alias_(exp.Window(this=exp.Count(this=exp.Star())), _N))
    if _wrapped(node):
        body = exp.select("*", *windows).from_(body.subquery(_SRC + "_src"))
    else:
        body.set("expressions", [*body.expressions, *windows])

    rows = (
        exp.Floor(
            this=exp.Div(
                this=exp.Mul(
                    this=exp.cast(exp.Literal.number(count), "DOUBLE"), expression=exp.column(_N)
                ),
                expression=exp.Literal.number(100),
            )
        )
        if percent
        else exp.Literal.number(count)
    )
    last = exp.Add(this=exp.Literal.number(skip), expression=rows)
    source = body.subquery(_SRC)
    # The kept rows by position; with ties, the peers of the last of them join them.
    skipped = exp.GT(this=exp.column(_RN), expression=exp.Literal.number(skip))
    if not ties:
        cut = exp.LTE(this=exp.column(_RN), expression=last)
    elif skip == 0 and not percent:
        cut = exp.LTE(this=exp.column(_RK), expression=rows)  # the n-th row's rank is <= n
    else:
        cutoff = exp.Window(
            this=exp.Max(
                this=exp.Case(
                    ifs=[
                        exp.If(
                            this=exp.and_(
                                skipped.copy(), exp.LTE(this=exp.column(_RN), expression=last)
                            ),
                            true=exp.column(_RK),
                        )
                    ]
                )
            )
        )
        hidden = [exp.column(c) for c in (_RN, _RK) + ((_N,) if percent else ())]
        cutting = exp.select("*", *hidden, exp.alias_(cutoff, _CUT)).from_(source)
        source = cutting.subquery(_SRC + "_ranked")
        cut = exp.LTE(this=exp.column(_RK), expression=exp.column(_CUT))
    where = exp.and_(skipped, cut)
    return exp.select("*").from_(source).where(where).order_by(exp.Ordered(this=exp.column(_RN)))


def _constant(node) -> float | None:
    """The numeric value of a literal count or offset; None when absent."""
    if node is None or isinstance(node, exp.Identifier):
        return None
    try:
        value = float(node.to_py())
    except (TypeError, ValueError, AttributeError):
        raise PlanError(
            f"a LIMIT / FETCH / OFFSET count must be a constant number, got {node.sql()}"
        ) from None
    return int(value) if value.is_integer() else value


def _wrapped(node) -> bool:
    """Whether the ranking windows must sit over the query's output rather than inside it."""
    return isinstance(node, exp.Union) or bool(node.args.get("distinct"))


def _order_keys(node, order) -> list:
    """The ORDER BY keys as window order keys, aliases and positions resolved."""
    projections = _projections(node)
    names = {_output_name(p): p for p in projections}
    keys = []
    for ordered in order.expressions:
        key = ordered.this
        if isinstance(key, exp.Literal) and not key.is_string:
            index = int(key.this)
            if not 1 <= index <= len(projections) or isinstance(projections[index - 1], exp.Star):
                raise PlanError(f"ORDER BY position {index} is not a selected column")
            target = projections[index - 1]
        elif isinstance(key, exp.Column) and not key.table and key.name in names:
            target = names[key.name]
        elif _wrapped(node):
            raise NotImplementedError(
                "LIMIT PERCENT / WITH TIES over a DISTINCT or set-operation query needs its "
                f"ORDER BY to name selected columns; {key.sql()} is not one"
            )
        else:
            keys.append(ordered.copy())
            continue
        if _wrapped(node):
            resolved = exp.column(_output_name(target))
        else:
            resolved = (target.this if isinstance(target, exp.Alias) else target).copy()
        keys.append(_with(ordered, resolved))
    return keys


def _with(ordered, key):
    """A copy of `ordered` sorting on `key` instead."""
    out = ordered.copy()
    out.set("this", key)
    return out


def _projections(node) -> list:
    """The select list whose columns the query outputs (the leftmost one of a set op)."""
    while isinstance(node, exp.Union):
        node = node.this
    return list(node.expressions) if isinstance(node, exp.Select) else []


def _output_name(p) -> str:
    """The name a select item's column is output under."""
    return p.alias_or_name
