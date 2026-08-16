"""Verify that a query which asked for an order actually got one.

The correctness gate in `harness` compares results as **row multisets**: it sorts both
sides on every column before comparing, because engines are free to return an unordered
result in any order and a positional comparison would fail on that alone. That is right for
everything except the queries that asked for an order — and for those it is a hole big
enough to drive a wrong answer through:

    >>> results_match(pa.table({"c": ["a", "b", "c"]}), pa.table({"c": ["c", "a", "b"]}))
    (True, 'ok')

Both sides get sorted, so an engine that ran `ORDER BY l_comment` and returned the rows
untouched passes the gate and is then *timed*, and the row reads as a win on a sort it never
performed. The operator mix has four such cases, and TPC-H, TPC-DS and ClickBench are full of
`ORDER BY ... LIMIT`. This is the failure `CLAUDE.md` names — "never assert a sort with an
order-independent comparison" — sitting in the harness that judges the engine.

So a case that carries an `ORDER BY` is additionally checked for **monotonicity in the
result's own order**, per engine, before its timing is trusted.

## What it checks, and what it deliberately does not

It verifies the result is sorted by the keys the query named, lexicographically, in the
stated direction. It does **not** compare positions between engines, and it does not check
where nulls sit. Both omissions are to make a false failure impossible:

* **Ties are unspecified.** `ORDER BY c` over duplicate `c` leaves the tied rows in any
  order, so two correct engines legitimately disagree on their arrangement. Monotonicity of
  the keys is the property the query actually guarantees.
* **Null placement differs by engine.** DuckDB defaults to NULLS LAST for both directions,
  PostgreSQL to NULLS LAST for `ASC` and NULLS FIRST for `DESC`, and the benchmark queries
  mostly do not say. Any adjacent pair where the deciding key is null on either side is
  therefore skipped rather than judged.
* **A key it cannot resolve to an output column stops the check** at that position. A
  partial check — the primary key verified, a computed secondary key not — is strictly more
  than none and still cannot fail a correct result.

The consequence to keep in mind when reading a green run: this proves an engine's result *is
ordered*, not that it is ordered identically to another engine's.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from .names import canonical_column_name

# One `ORDER BY` term: the output column it names (as a 1-based ordinal, or the term's text
# for a by-name match) and whether it descends.
OrderKey = tuple[str | int, bool]


def order_keys_of(query: str) -> list[OrderKey]:
    """The outermost `ORDER BY` terms of `query`, or `[]` when it has none.

    Only the outermost order matters: a subquery's `ORDER BY` constrains nothing about the
    statement's result, and every engine is free to discard it.

    Args:
        query: The SQL the case runs.

    Returns:
        One `(term, descending)` per key, in order. `term` is an `int` for a positional
        `ORDER BY 3` and the term's canonicalized text otherwise.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:  # pragma: no cover - sqlglot ships with batcher
        return []
    try:
        tree = sqlglot.parse_one(query)
    except Exception:
        # An unparseable query is one this check has nothing to say about; the multiset
        # gate still runs. Never fail a case over the *checker's* limits.
        return []
    order = tree.args.get("order")
    if order is None:
        return []
    keys: list[OrderKey] = []
    for term in order.expressions:
        inner = term.this
        descending = bool(term.args.get("desc"))
        if isinstance(inner, exp.Literal) and inner.is_int:
            keys.append((int(inner.this), descending))
        else:
            keys.append((canonical_column_name(inner.sql()), descending))
    return keys


def order_violation(table: pa.Table, keys: list[OrderKey]) -> str | None:
    """The first place `table` is not ordered by `keys`, or `None` when it is.

    Args:
        table: One engine's result, in the order it produced.
        keys: The query's `ORDER BY` terms, from `order_keys_of`.

    Returns:
        A message naming the key and the first offending row pair, or `None`.
    """
    if table.num_rows < 2 or not keys:
        return None
    resolved = _resolve(table, keys)
    if not resolved:
        return None
    # `violation[i]` is true when row `i` must not precede row `i+1`: some key decides the
    # pair (every earlier key ties) and decides it the wrong way round. Built key by key so
    # the whole test is Arrow kernels over `num_rows - 1` elements — a row-wise walk would
    # cost more than the sort it is checking.
    ties: pa.Array | None = None
    violation: pa.Array | None = None
    for name, descending in resolved:
        col = table.column(name)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        if pa.types.is_null(col.type):
            # An all-null key column: every pair ties on it (both sides null), and arrow has
            # no comparison kernel for the `null` type to say so. Nothing decided here, so
            # the next key judges the pair exactly as it would after a real tie.
            continue
        lhs, rhs = col.slice(0, len(col) - 1), col.slice(1)
        # A null on either side leaves the pair undecided rather than wrong: engines place
        # nulls differently and these queries mostly do not say where they belong.
        wrong = pc.fill_null(pc.less(lhs, rhs) if descending else pc.greater(lhs, rhs), False)
        # Tied *only* when both sides are known equal, or both are null. One null is not a
        # tie: the pair is decided at this key, by a placement rule the query did not state,
        # so no later key may judge it either. Calling it a tie (which is what filling the
        # null comparison with `True` does) hands the pair to the next key and reports the
        # engine's own null placement as unsortedness — it failed nine TPC-DS queries that
        # way, for DuckDB as loudly as for Batcher, which is how the bug announced itself.
        equal = pc.or_(
            pc.fill_null(pc.equal(lhs, rhs), False),
            pc.and_(lhs.is_null(), rhs.is_null()),
        )
        if violation is None:
            violation, ties = wrong, equal
        else:
            violation = pc.or_(violation, pc.and_(ties, wrong))
            ties = pc.and_(ties, equal)
    if violation is None:
        return None  # every resolved key was an all-null column: nothing is decided
    bad = pc.indices_nonzero(violation)
    if len(bad) == 0:
        return None
    row = bad[0].as_py()

    # Name the key that actually decided the pair, not the leading one: on a multi-key sort
    # the leading key is usually tied there, and reporting it reads as a contradiction. The
    # deciding key is the first whose two values are both present and the wrong way round.
    def out_of_order(name: str, descending: bool) -> bool:
        a, b = table.column(name)[row].as_py(), table.column(name)[row + 1].as_py()
        if a is None or b is None:
            return False
        return b > a if descending else a > b

    name, descending = next(
        ((n, d) for n, d in resolved if out_of_order(n, d)),
        resolved[0],
    )
    direction = "DESC" if descending else "ASC"
    return (
        f"not ordered by {name!r} {direction}: row {row} is "
        f"{table.column(name)[row].as_py()!r} before {table.column(name)[row + 1].as_py()!r}"
    )


def _resolve(table: pa.Table, keys: list[OrderKey]) -> list[tuple[str, bool]]:
    """`keys` mapped onto `table`'s columns, stopping at the first one that cannot be.

    A key that names a computed expression the SELECT aliased to something else is not
    resolvable from the result alone, and the keys after it are only meaningful *given* it —
    so the prefix is what gets checked.
    """
    by_name = {canonical_column_name(n): n for n in table.column_names}
    out: list[tuple[str, bool]] = []
    for term, descending in keys:
        if isinstance(term, int):
            if not 1 <= term <= table.num_columns:
                break
            out.append((table.column_names[term - 1], descending))
            continue
        column = by_name.get(term)
        if column is None:
            break
        out.append((column, descending))
    return out
