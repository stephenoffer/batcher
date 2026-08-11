"""Module-level expression constructors (the user-facing entry points).

`col`, `lit`, `when`, `coalesce`, `nullif`, `atan2`, `greatest`, `least`, and
`count` build expression trees out of the node classes in `core`. These are the
free functions users call directly (e.g. `col("x")`, `when(c).then(v)`).
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import (
    AggExpr,
    Coalesce,
    Expr,
    IntoExpr,
    Lit,
    Math2Expr,
    _wrap,
)
from batcher.plan.expr_ir.nodes import (
    Array,
    CaseBuilder,
    Col,
    Greatest,
    HashRows,
    Least,
    NullIf,
)


def when(cond: Expr) -> CaseBuilder:
    """Begin a CASE expression.

    Returns a builder you chain with ``.then(value)`` and finish with
    ``.otherwise(default)``; add further ``.when(...).then(...)`` pairs for more
    branches. The first matching condition wins, evaluated row by row.

    Args:
        cond: A boolean expression selecting the rows this branch applies to.

    Returns:
        A `CaseBuilder`; call ``.then(...).otherwise(...)`` to produce the expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 0, 5]})
            >>> grade = bt.when(bt.col("x") > 0).then(bt.lit("pos")).otherwise(bt.lit("non-pos"))
            >>> ds.select(grade=grade).to_pydict()
            {'grade': ['non-pos', 'non-pos', 'pos']}
    """
    return CaseBuilder().when(cond)


def array(*elements: IntoExpr) -> Array:
    """A list literal built per row from the element expressions (SQL ``ARRAY[...]``).

    Each output row is a list of the per-row element values, coerced to a common
    type. Use it to pack several columns into one list column — a feature vector,
    an embedding, or a set passed to a list operation.

    A single list or tuple is accepted as the elements themselves, so a query vector
    already held in a Python list needs no unpacking. That spelling is the natural one
    for the vector-distance kernels — ``col("emb").list.cosine_similarity(array(q))`` —
    and it used to build an ``Array`` whose one element was the list, failing much later
    inside ``to_ir`` with ``unsupported literal type: list``: a message naming neither
    this function nor the remedy, on a traceback pointing at ``collect()``. There is no
    competing meaning, because a nested list has no literal spelling here.

    Args:
        *elements: One or more expressions, one per list position, or a single list or
            tuple holding them.

    Returns:
        An expression producing a `List` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1], "b": [2]})
            >>> ds.select(pair=bt.array(bt.col("a"), bt.col("b"))).to_pydict()
            {'pair': [[1, 2]]}

            >>> ds = bt.from_pydict({"emb": [[1.0, 0.0]]})
            >>> sim = bt.col("emb").list.cosine_similarity(bt.array([1.0, 0.0]))
            >>> ds.select(sim=sim).to_pydict()
            {'sim': [1.0]}
    """
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
        elements = tuple(elements[0])
    if not elements:
        raise ValueError("array() requires at least one element")
    return Array([_wrap(e) for e in elements])


def coalesce(*exprs: IntoExpr) -> Coalesce:
    """First non-null among the arguments, per row (SQL ``COALESCE``).

    Evaluates the arguments left to right and returns the first that is not null,
    or null if all are. The usual use is a fallback for a nullable column, e.g.
    ``coalesce(col("discount"), lit(0))`` to treat a missing discount as zero.

    Args:
        *exprs: One or more expressions, tested in order.

    Returns:
        An expression equal to the first non-null argument.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, None, 3], "b": [10, 20, 30]})
            >>> ds.select(c=bt.coalesce(bt.col("a"), bt.col("b"))).to_pydict()
            {'c': [1, 20, 3]}
    """
    if not exprs:
        raise ValueError("coalesce() requires at least one argument")
    return Coalesce([_wrap(e) for e in exprs])


def nullif(left: IntoExpr, right: IntoExpr) -> NullIf:
    """Null where ``left == right``, else ``left`` (SQL ``NULLIF``).

    Returns null when the two arguments are equal, otherwise the left value. Useful
    for turning a sentinel into a real null (``nullif(col("x"), lit(-1))``) or
    guarding a divisor against zero (``a / nullif(b, lit(0))`` yields null, not an
    error, when ``b`` is 0).

    Args:
        left: The value returned when the two differ.
        right: The value that, when equal to ``left``, produces null.

    Returns:
        An expression that is null on equality, else ``left``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 5, 5]})
            >>> ds.select(r=bt.nullif(bt.col("x"), bt.lit(5))).to_pydict()
            {'r': [1, None, None]}
    """
    return NullIf(_wrap(left), _wrap(right))


def atan2(y: IntoExpr, x: IntoExpr) -> Math2Expr:
    """Two-argument arctangent of ``y / x`` (→ Float64).

    Computes the angle of the point ``(x, y)`` from the positive x-axis, using the
    signs of both arguments to place it in the correct quadrant, so the result
    spans the full ``[-π, π]`` range (unlike single-argument ``atan``).

    Args:
        y: The ordinate (numerator).
        x: The abscissa (denominator).

    Returns:
        A Float64 expression of the angle in radians.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0], "x": [1.0]})
            >>> ds.select(r=bt.atan2(bt.col("y"), bt.col("x"))).to_pydict()
            {'r': [0.0]}
    """
    return Math2Expr("atan2", _wrap(y), _wrap(x))


def hash_rows(*exprs: IntoExpr, seed: int = 0) -> HashRows:
    """A deterministic 64-bit hash of the given values, per row → Int64.

    Typed rather than textual: an integer hashes its bits, a float its canonicalized
    IEEE bits (so ``-0.0`` and ``0.0`` agree, and every NaN agrees), a string its UTF-8
    bytes. That makes it independent of how a float renders, and far cheaper than
    hashing ``cast(col, "string")``. Order-sensitive across `exprs`, and null is a
    distinct value — ``hash_rows(1, None)`` and ``hash_rows(None, 1)`` differ, and
    neither collides with ``hash_rows(1, 1)``.

    The digest is stable across partitions, runs, machines and Batcher versions, which
    is what lets it key a reproducible train/test split, a surrogate key, or a hash
    bucket. Two rows that compare equal always hash equally; two that differ may (very
    rarely) collide, as with any 64-bit hash.

    Args:
        *exprs: The values to hash, in order. At least one is required.
        seed: Changes the digest; the same seed reproduces it.

    Returns:
        An Int64 expression — the row's digest.

    Raises:
        PlanError: If no expressions are given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 1, 2], "b": ["x", "x", "x"]})
            >>> h = ds.select(h=bt.hash_rows(bt.col("a"), bt.col("b"))).to_pydict()["h"]
            >>> h[0] == h[1], h[0] == h[2]
            (True, False)

            >>> # Deterministic bucketing: 10 stable buckets, partition-independent.
            >>> ds.select(bucket=bt.hash_rows(bt.col("a")).abs() % 10).to_pydict()["bucket"]
            [9, 9, 5]
    """
    if not exprs:
        raise PlanError("hash_rows() requires at least one expression")
    return HashRows([_wrap(e) for e in exprs], int(seed))


def greatest(*exprs: IntoExpr) -> Greatest:
    """The largest argument per row, ignoring nulls (SQL ``GREATEST``).

    Compares the arguments value by value within each row and returns the maximum,
    skipping nulls; a row that is null in every argument yields null. This is a
    row-wise (horizontal) max across columns, not an aggregate down a column — for
    that, use ``col("x").max()`` inside ``agg``.

    Args:
        *exprs: One or more expressions to compare.

    Returns:
        An expression equal to the per-row maximum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(hi=bt.greatest(bt.col("a"), bt.col("b"))).to_pydict()
            {'hi': [4, 9]}
    """
    if not exprs:
        raise ValueError("greatest() requires at least one argument")
    return Greatest([_wrap(e) for e in exprs])


def least(*exprs: IntoExpr) -> Least:
    """The smallest argument per row, ignoring nulls (SQL ``LEAST``).

    The row-wise (horizontal) minimum across the given expressions, skipping nulls;
    an all-null row yields null. The counterpart to `greatest`.

    Args:
        *exprs: One or more expressions to compare.

    Returns:
        An expression equal to the per-row minimum.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1, 9], "b": [4, 2]})
            >>> ds.select(lo=bt.least(bt.col("a"), bt.col("b"))).to_pydict()
            {'lo': [1, 2]}
    """
    if not exprs:
        raise ValueError("least() requires at least one argument")
    return Least([_wrap(e) for e in exprs])


def col(name: str) -> Col:
    """Reference an input column by name.

    ``col`` is the starting point for almost every expression: it names a column in
    the dataset, and the operators (``+``, ``==``, ``&`` …) and methods (``.sum()``,
    ``.cast(...)``, ``.str.upper()`` …) on the result build the computation that
    runs in the Rust engine. It is lazy and does no work itself.

    Args:
        name: The name of an existing column.

    Returns:
        An expression that evaluates to that column's values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"price": [10, 20], "qty": [2, 3]})
            >>> ds.select(total=bt.col("price") * bt.col("qty")).to_pydict()
            {'total': [20, 60]}
    """
    return Col(name)


def count() -> AggExpr:
    """``COUNT(*)`` — the number of rows in each group.

    Use inside ``group_by(...).agg(...)`` to count rows per group, or with no
    grouping to count the whole dataset. It counts rows, not non-null values, so it
    takes no column; for non-null counts use ``col("x").count()``.

    Returns:
        An aggregate expression; pass it to ``.agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
            >>> ds.group_by("g").agg(n=bt.count()).sort("g").to_pydict()
            {'g': ['a', 'b'], 'n': [2, 1]}
    """
    return AggExpr("count_star", None)


def lit(value: int | float | bool | str) -> Lit:
    """A constant literal expression.

    Wraps a Python scalar so it can be combined with column expressions — a default
    in ``when(...).otherwise(bt.lit(0))``, an offset like ``bt.col("x") + bt.lit(1)``,
    or a fallback in ``coalesce(col("x"), bt.lit(0))``. Bare Python scalars are
    accepted in most places too; ``lit`` is the explicit form.

    Args:
        value: The constant value (int, float, bool, or str).

    Returns:
        An expression that evaluates to ``value`` on every row.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> ds.select(y=bt.col("x") + bt.lit(100)).to_pydict()
            {'y': [101, 102]}
    """
    return Lit(value)
