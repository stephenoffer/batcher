"""``ASOF JOIN`` lowering — the SQL spelling of `Dataset.join_asof`.

sqlglot carries ``ASOF`` in a `Join`'s ``method`` slot, the same slot as ``NATURAL``.
The translator only ever read that slot for ``NATURAL``, so an ``ASOF JOIN`` fell through
to the ordinary ``ON``-predicate path and ran as a **theta join**: every right row
satisfying the inequality was emitted, not the single nearest one. That is a wrong answer
rather than an error — ``t ASOF JOIN u ON t.id >= u.id`` returned 21 rows where DuckDB
returns 8 — which is why the method slot is read here.

The ``ON`` condition splits the same way DuckDB's does: every equality conjunct is an
exact-match ``by`` key, and the single inequality conjunct is the nearest-match key, with
its operator fixing the direction (``left >= right`` looks backward, ``left <= right``
looks forward).
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.parser.joins.theta import and_conjuncts
from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit

__all__ = ["asof_join", "is_asof"]

# The nearest-match operators, mapped to the direction they scan in. Only the inclusive
# forms appear: `AsofJoin.direction` is defined as "largest right <= left" / "smallest
# right >= left", so a *strict* `>` or `<` has no representation and is rejected below
# rather than silently answered with the inclusive match.
_DIRECTIONS = {exp.GTE: "backward", exp.LTE: "forward"}
_STRICT = (exp.GT, exp.LT)

# Marks a right row as present so an inner ASOF can drop the rows that matched nothing.
# A right column being null does not mean "unmatched" — the data may hold nulls — so the
# witness is added to the right relation rather than inferred from an output column.
_WITNESS = "__bc_asof_matched"


def is_asof(join) -> bool:
    """Whether this `Join` node is an ``ASOF JOIN``."""
    return (join.args.get("method") or "").upper() == "ASOF"


def _oriented(conj, left_cols: set[str], right_cols: set[str]) -> tuple[str, str]:
    """A comparison's ``(left column, right column)``, by which relation owns each name.

    ``ON u.id <= t.id`` is the same join as ``ON t.id >= u.id``; the caller has already
    normalized the operator, so all that remains is to bind each name to its relation.
    """
    a, b = conj.this, conj.expression
    if not (isinstance(a, exp.Column) and isinstance(b, exp.Column)):
        raise PlanError(
            "ASOF JOIN needs its inequality to compare two columns "
            f"(got {conj.sql()}); compute the expression in a subquery first"
        )
    if a.name in left_cols and b.name in right_cols:
        return a.name, b.name
    if b.name in left_cols and a.name in right_cols:
        return b.name, a.name
    raise PlanError(
        f"ASOF JOIN inequality {conj.sql()} does not compare one column from each side "
        f"(left has {sorted(left_cols)}, right has {sorted(right_cols)})"
    )


def _split(on, left_cols: set[str], right_cols: set[str]) -> tuple[list, str, str, str]:
    """Split the ``ON`` into exact-match `by` pairs plus the one nearest-match key."""
    pairs: list[tuple[str, str]] = []
    match: tuple[str, str, str] | None = None
    for conj in and_conjuncts(on):
        if isinstance(conj, exp.EQ):
            pairs.append(_oriented(conj, left_cols, right_cols))
            continue
        if isinstance(conj, _STRICT):
            raise PlanError(
                f"ASOF JOIN with the strict inequality {conj.sql()} is not supported; "
                "the nearest-match key is inclusive, so use >= or <="
            )
        direction = _DIRECTIONS.get(type(conj))
        if direction is None:
            raise PlanError(
                f"ASOF JOIN condition {conj.sql()} must be an equality (an exact-match "
                "key) or a >= / <= inequality (the nearest-match key)"
            )
        if match is not None:
            raise PlanError(
                "ASOF JOIN takes exactly one nearest-match inequality; "
                f"got both {match[2]} and {conj.sql()}"
            )
        left_on, right_on = _oriented(conj, left_cols, right_cols)
        # `_oriented` may have swapped the operands to bind them to their relations, which
        # flips which way the comparison reads. Re-derive the direction from the oriented
        # pair rather than from the operator as typed.
        if conj.this.name != left_on:
            direction = "forward" if direction == "backward" else "backward"
        match = (left_on, right_on, conj.sql(), direction)  # type: ignore[assignment]
    if match is None:
        raise PlanError(
            "ASOF JOIN requires a nearest-match inequality (>= or <=) in its ON "
            "condition; an ON with equalities alone is an ordinary join"
        )
    left_on, right_on, _, direction = match  # type: ignore[misc]
    return pairs, left_on, right_on, direction


def asof_join(left: Dataset, right: Dataset, join, how: str) -> Dataset:
    """Lower one ``ASOF JOIN`` clause onto `Dataset.join_asof`.

    Args:
        left: The relation accumulated so far, which drives the match.
        right: The relation being joined in.
        join: The sqlglot `Join` node, for its ``ON`` condition.
        how: ``"inner"`` for ``ASOF JOIN`` (unmatched left rows are dropped) or
            ``"left"`` for ``ASOF LEFT JOIN`` (they are kept, null-extended).

    Returns:
        The joined dataset.
    """
    on = join.args.get("on")
    if on is None:
        raise PlanError(
            "ASOF JOIN requires an ON condition naming the nearest-match key "
            "(e.g. ON left.ts >= right.ts)"
        )
    if how not in {"inner", "left"}:
        raise PlanError(
            f"ASOF {how.upper()} JOIN is not supported; ASOF matches each left row, so "
            "only ASOF JOIN (inner) and ASOF LEFT JOIN are defined"
        )
    pairs, left_on, right_on, direction = _split(on, set(left.columns), set(right.columns))

    # `join_asof` *coalesces* its key columns: the output carries the left side's and drops
    # the right's. That is right for the DataFrame API, and wrong for SQL's ON form, where
    # both keys stay separate columns — so `SELECT o.id, q.id FROM o ASOF JOIN q ON o.id >=
    # q.id` died with `references unknown column 'q__id'`. Each right key is copied to a
    # shadow the join cannot coalesce and restored afterwards, the same shape
    # `_join_keeping_both_keys` uses for the ordinary ON join.
    shadows = {f"__bc_asof_k{i}": k for i, k in enumerate([right_on, *(q for _, q in pairs)])}
    extra = {s: col(k) for s, k in shadows.items()}
    if how == "inner":
        # Carry a non-null witness through the (left-style) asof join so the rows that
        # matched nothing can be dropped; a right column being NULL does not mean
        # "unmatched", since the data may hold nulls.
        extra[_WITNESS] = lit(True)

    joined = left.join_asof(
        right.with_columns(**extra),
        left_on=left_on,
        right_on=right_on,
        left_by=[p for p, _ in pairs],
        right_by=[q for _, q in pairs],
        direction=direction,
    )
    if how == "inner":
        joined = joined.filter(col(_WITNESS).is_not_null()).drop(_WITNESS)
    # A key whose name the left side already carries has no column of its own to be
    # restored into — that is the same-name case the disambiguator normally renames apart,
    # and where it could not reach, the coalesced column is the only answer available.
    restore = {k: col(s) for s, k in shadows.items() if k not in joined.columns}
    joined = joined.with_columns(**restore) if restore else joined
    return joined.drop(*shadows)
