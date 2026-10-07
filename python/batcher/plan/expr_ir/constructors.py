"""Module-level expression constructors (the user-facing entry points).

`col`, `lit`, `when`, `coalesce`, `nullif`, `greatest`, `least`, and
`count` build expression trees out of the node classes in `core`. These are the
free functions users call directly (e.g. `col("x")`, `when(c).then(v)`).
"""

from __future__ import annotations

import decimal
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import (
    AggExpr,
    Coalesce,
    Expr,
    IntoExpr,
    Lit,
    _col_or_expr,
    _is_null_literal,
    _wrap,
)
from batcher.plan.expr_ir.nodes import (
    Array,
    CaseBuilder,
    Col,
    Greatest,
    HashRows,
    Least,
    MakeStruct,
    NullIf,
)
from batcher.plan.types.registry import dtype_name, resolve_dtype

if TYPE_CHECKING:
    import pyarrow as pa


def when(cond: Expr) -> CaseBuilder:
    """Begin a CASE expression.

    Returns a builder you chain with ``.then(value)`` and optionally finish with
    ``.otherwise(default)``; add further ``.when(...).then(...)`` pairs for more
    branches. The first matching condition wins, evaluated row by row.

    Without ``.otherwise`` (or with ``.otherwise(None)``) a row no branch matches is
    NULL, as SQL's ``CASE WHEN ... END`` is. The NULL takes the type of the first
    non-null branch value.

    **Every branch is evaluated on every row**, and the condition only picks which
    result each row keeps. So a ``when`` cannot guard a branch that would *fail* on the
    rows it excludes: ``when(col("s") != "x").then(col("s").cast("int64"))`` still raises
    on ``"x"``, where DuckDB, which evaluates lazily, returns NULL. Make the branch itself
    safe instead, with `Expr.try_cast` (null on a failed cast) or `Expr.safe_divide`. A
    branch that merely produces a value you discard, such as ``sqrt`` of a negative, is
    fine.

    Args:
        cond: A boolean expression selecting the rows this branch applies to.

    Returns:
        A `CaseBuilder`, usable as an expression once it has a ``.then(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 0, 5]})
            >>> grade = bt.when(bt.col("x") > 0).then(bt.lit("pos")).otherwise(bt.lit("non-pos"))
            >>> ds.select(grade=grade).to_pydict()
            {'grade': ['non-pos', 'non-pos', 'pos']}

            >>> ds.select(pos=bt.when(bt.col("x") > 0).then(bt.col("x"))).to_pydict()
            {'pos': [None, None, 5]}

            >>> s = bt.from_pydict({"s": ["1", "x"]})
            >>> safe = bt.when(bt.col("s") != "x").then(bt.col("s").try_cast("int64"))
            >>> s.select(n=safe).to_pydict()
            {'n': [1, None]}
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
        raise PlanError("array() requires at least one element")
    return Array([_wrap(e) for e in elements])


def coalesce(*exprs: IntoExpr) -> Coalesce:
    """First non-null among the arguments, per row (SQL ``COALESCE``).

    Evaluates the arguments left to right and returns the first that is not null,
    or null if all are. The usual use is a fallback for a nullable column, e.g.
    ``coalesce(col("discount"), lit(0))`` to treat a missing discount as zero.

    A bare string names a **column**, as it does in Polars. Spell a string constant
    ``bt.lit("...")``. A ``None`` argument (or an untyped ``bt.lit(None)``) is dropped,
    because ``COALESCE(x, NULL)`` is ``x``: keeping it would only give the result an Int64
    NULL to reconcile with a string or date column.

    Args:
        *exprs: One or more expressions or column names, tested in order.

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
        raise PlanError("coalesce() requires at least one argument")
    kept = [e for e in exprs if not _is_null_literal(e)]
    if not kept:
        return Coalesce([null()])
    return Coalesce([_col_or_expr(e) for e in kept])


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


def hash_rows(*exprs: IntoExpr, seed: int = 0, algorithm: str = "batcher") -> HashRows:
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

    `algorithm` reproduces another engine's digest bit for bit, for a ported job whose
    stored keys or buckets must not move:

    - ``"murmur3"`` is Spark ``hash(...)``: 32-bit Murmur3 chained across the inputs from
      `seed` (Spark's is 42), a null input leaving the hash unchanged, the Int32 result
      sign-extended. Spark hashes an ``int`` column as 4 bytes and a ``bigint`` as 8, so
      cast to ``int32`` where the Spark column was an ``IntegerType``.
    - ``"iceberg"`` is the Iceberg bucket-transform hash of one value (standard Murmur3,
      integers and dates as 8-byte longs); a null stays null. See ``Expr.hash_bucket``.
    - ``"xxhash3"`` is Daft's default ``hash``: XXH3-64 of one value with `seed`, read
      back as signed. A null hashes like an empty input, as in Daft.

    Args:
        *exprs: The values to hash, in order. At least one is required.
        seed: Changes the digest; the same seed reproduces it.
        algorithm: ``"batcher"`` (the default), ``"murmur3"``, ``"iceberg"`` or
            ``"xxhash3"``.

    Returns:
        An Int64 expression — the row's digest.

    Raises:
        PlanError: If no expressions are given, `algorithm` is unknown, or a
            single-value algorithm gets more than one expression.

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
    if algorithm not in _HASH_ALGORITHMS:
        raise PlanError(
            f"hash_rows(): algorithm must be one of {sorted(_HASH_ALGORITHMS)}, got {algorithm!r}"
        )
    if algorithm in ("iceberg", "xxhash3") and len(exprs) != 1:
        raise PlanError(f"hash_rows(algorithm={algorithm!r}) hashes exactly one expression")
    wire = None if algorithm == "batcher" else algorithm
    return HashRows([_wrap(e) for e in exprs], int(seed), wire)


#: The digests `hash_rows` computes; mirrors `bc_expr::HashAlgorithm`.
_HASH_ALGORITHMS: Final = frozenset({"batcher", "iceberg", "murmur3", "xxhash3"})


def greatest(*exprs: IntoExpr) -> Greatest:
    """The largest argument per row, ignoring nulls (SQL ``GREATEST``).

    Compares the arguments value by value within each row and returns the maximum,
    skipping nulls; a row that is null in every argument yields null. This is a
    row-wise (horizontal) max across columns, not an aggregate down a column — for
    that, use ``col("x").max()`` inside ``agg``.

    A bare string names a **column**, as it does in Polars. Spell a string constant
    ``bt.lit("...")``.

    Args:
        *exprs: One or more expressions or column names to compare.

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
        raise PlanError("greatest() requires at least one argument")
    return Greatest([_col_or_expr(e) for e in exprs])


def least(*exprs: IntoExpr) -> Least:
    """The smallest argument per row, ignoring nulls (SQL ``LEAST``).

    The row-wise (horizontal) minimum across the given expressions, skipping nulls;
    an all-null row yields null. The counterpart to `greatest`.

    A bare string names a **column**, as it does in Polars. Spell a string constant
    ``bt.lit("...")``.

    Args:
        *exprs: One or more expressions or column names to compare.

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
        raise PlanError("least() requires at least one argument")
    return Least([_col_or_expr(e) for e in exprs])


def col(name: str | pa.DataType | Iterable[str | pa.DataType], *more: str | pa.DataType) -> Expr:
    """Reference an input column by name, or several columns at once.

    ``col`` is the starting point for almost every expression: it names a column in
    the dataset, and the operators (``+``, ``==``, ``&`` …) and methods (``.sum()``,
    ``.cast(...)``, ``.str.upper()`` …) on the result build the computation that
    runs in the Rust engine. It is lazy and does no work itself.

    Given more than one name, a list of names, a regular expression wrapped in ``^...$``,
    or an Arrow type (or several), it is a column selector, as Polars' ``col`` is: it
    expands to one expression per matched column when a projection is built
    (``select``, ``with_columns``, ``group_by().agg``), so ``col("a", "b") * 2`` doubles
    both. Named columns expand in the order given and must all exist; a pattern or a type
    matches in the dataset's column order. ``bt.matches`` and ``bt.by_dtype`` are the same
    selectors under their own names.

    Args:
        name: A column name, a ``^...$`` pattern, an Arrow type, or a list of them.
        *more: Further names, patterns or types, selecting several columns.

    Returns:
        A column expression, or a selector when several columns may match.

    Raises:
        PlanError: If names and Arrow types are mixed, or nothing is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow as pa
            >>> ds = bt.from_pydict({"price": [10, 20], "qty": [2, 3], "sku": ["a", "b"]})
            >>> ds.select(total=bt.col("price") * bt.col("qty")).to_pydict()
            {'total': [20, 60]}
            >>> ds.select(bt.col("qty", "price") * 10).to_pydict()
            {'qty': [20, 30], 'price': [100, 200]}
            >>> ds.select(bt.col("^p.*$")).columns, ds.select(bt.col(pa.string())).columns
            (['price'], ['sku'])
    """
    items = [*_col_items(name), *(m for x in more for m in _col_items(x))]
    if not items:
        raise PlanError("col() requires a column name")
    if len(items) == 1 and isinstance(items[0], str) and not _is_pattern(items[0]):
        return Col(items[0])
    return _col_selector(items)


def _col_items(value: Any) -> list[Any]:
    """One `col` argument as a flat list: a name, pattern or type, or a list of them."""
    # A plain name is the overwhelmingly common call, and answering it before touching
    # pyarrow keeps `bt.col` from importing pyarrow at all.
    if isinstance(value, str):
        return [value]
    import pyarrow as pa

    if isinstance(value, pa.DataType):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    raise PlanError(f"col() takes column names or Arrow types, got {type(value).__name__}")


def _is_pattern(name: str) -> bool:
    """Polars reads a name wrapped in ``^...$`` as a regular expression, and so does `col`."""
    return len(name) >= 2 and name.startswith("^") and name.endswith("$")


def _col_selector(items: list[Any]) -> Expr:
    """The selector a multi-column `col` stands for."""
    import pyarrow as pa

    from batcher.plan.expr_ir.selectors import by_dtype, matches
    from batcher.plan.expr_ir.selectors.core import _named_columns

    types = [i for i in items if isinstance(i, pa.DataType)]
    if types:
        if len(types) != len(items):
            raise PlanError("col() takes either column names or Arrow types, not both")
        return by_dtype(*types)
    if not any(_is_pattern(i) for i in items):
        return _named_columns(tuple(items))
    selector = None
    for item in items:
        part = matches(item) if _is_pattern(item) else _named_columns((item,))
        selector = part if selector is None else selector | part
    return selector


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


#: Arrow type names a literal may be given. `lit(..., dtype=...)` validates against this,
#: plus any ``decimal(p,s)``, before building, so a typo raises here rather than deep
#: inside `Cast`.
_NULL_DTYPES: Final = (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "bool", "string", "date", "time", "timestamp",
)  # fmt: skip


def _literal_dtype(dtype: str | None, func: str) -> str | None:
    """`dtype` checked against the names a literal may take, or a `PlanError` naming them.

    A ``decimal(p,s)`` is accepted alongside the fixed names, because a decimal literal is
    the one way to put an *exact* fractional constant into a query: a Python float is
    already inexact before Batcher sees it.
    """
    if dtype is None or dtype in _NULL_DTYPES:
        return dtype
    import pyarrow as pa

    spec = dtype.replace(" ", "").lower()
    resolved = resolve_dtype(spec) if spec.startswith("decimal(") else None
    if resolved is None or not pa.types.is_decimal(resolved):
        raise PlanError(
            f"{func}: unknown dtype {dtype!r}; expected one of {', '.join(_NULL_DTYPES)}, "
            "or 'decimal(p,s)'"
        )
    return spec


def lit(
    value: int
    | float
    | bool
    | str
    | decimal.Decimal
    | list[Any]
    | tuple[Any, ...]
    | dict[str, Any]
    | None,
    dtype: str | None = None,
) -> Expr:
    """A constant literal expression, or a typed NULL when `value` is None.

    Wraps a Python scalar so it can be combined with column expressions — a default
    in ``when(...).otherwise(bt.lit(0))``, an offset like ``bt.col("x") + bt.lit(1)``,
    or a fallback in ``coalesce(col("x"), bt.lit(0))``. Bare Python scalars are
    accepted in most places too; ``lit`` is the explicit form.

    ``lit(None)`` is the NULL literal. The JSON IR has no untyped null — `bc_expr::Literal`
    carries Int/Float/Bool/Str/Timestamp/Date and nothing else — so a null has to be given
    a type here, and it is built as ``nullif(1, 1)`` (null on every row, Int64) cast to
    `dtype`. Comparisons, `eq_missing`, `fill_null` and `coalesce` recognise the untyped
    form and answer without needing its type, so ``col("s") == bt.lit(None)`` is NULL on a
    string column too; elsewhere, give it a `dtype`.

    A list or tuple is a list literal (``bt.array`` of its elements) and a dict is a struct
    literal (``bt.struct`` of its items), each element a literal in turn. A
    `decimal.Decimal` is a double unless `dtype` is ``"decimal(p,s)"``, in which case it is
    exact: the value travels as its decimal text and is cast, never through a float. Any
    other value raises here, where it was written, rather than at `collect()`.

    Args:
        value: The constant value (int, float, bool, str, date/time/datetime,
            ``Decimal``, list, tuple or dict), or None for a NULL.
        dtype: Arrow type name for the literal, or ``"decimal(p,s)"``. Required only to
            type a NULL as something other than Int64; on a non-null value it is an
            explicit cast. Not accepted with a list or dict.

    Returns:
        An expression that evaluates to `value` on every row.

    Raises:
        PlanError: If `dtype` is not a type name a literal can take, or `value` has no
            literal form.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from decimal import Decimal
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> ds.select(y=bt.col("x") + bt.lit(100)).to_pydict()
            {'y': [101, 102]}

            >>> ds.select(y=bt.lit(None, dtype="string")).to_pydict()
            {'y': [None, None]}

            >>> ds.limit(1).select(v=bt.lit([1, 2]), s=bt.lit({"a": 1})).to_pydict()
            {'v': [[1, 2]], 's': [{'a': 1}]}

            >>> ds.limit(1).select(d=bt.lit(Decimal("1.25"), dtype="decimal(10,2)")).to_pydict()
            {'d': [Decimal('1.25')]}
    """
    func = "lit()"
    if isinstance(value, (list, tuple, dict)):
        if dtype is not None:
            raise PlanError(
                f"{func}: dtype= applies to a scalar literal, not a {type(value).__name__}; "
                "give each element its own lit(..., dtype=...)"
            )
        return _nested_literal(value)
    dtype = _literal_dtype(dtype, func)
    if value is None:
        return null(dtype)
    if isinstance(value, decimal.Decimal) and dtype is not None and dtype.startswith("decimal("):
        if not value.is_finite():
            raise PlanError(f"{func}: {value!r} has no decimal representation")
        return Lit(format(value, "f"))._cast(dtype, try_cast=False)
    typed = Lit(value)
    typed.to_ir()  # validate now: an unsupported value raises here, not at collect()
    return typed._cast(dtype, try_cast=False) if dtype is not None else typed


def _nested_literal(value: list[Any] | tuple[Any, ...] | dict[str, Any]) -> Expr:
    """A list literal as `array(...)` and a dict literal as `struct(...)`, recursively."""
    if isinstance(value, dict):
        if not value:
            raise PlanError("lit(): an empty dict has no struct type; give it at least one field")
        bad = [k for k in value if not isinstance(k, str)]
        if bad:
            raise PlanError(f"lit(): a struct literal's keys must be strings, got {bad[0]!r}")
        return MakeStruct([(k, lit(v)) for k, v in value.items()])
    if not value:
        raise PlanError(
            "lit(): an empty list has no element type; build it from a typed column instead"
        )
    return Array([lit(v) for v in value])


def null(dtype: str | None = None) -> Expr:
    """A NULL literal of `dtype` (Int64 when unspecified) — the neutral spelling of SQL NULL.

    Both front-ends need this and neither can hold it: the IR has no untyped null, so a
    NULL has to be *constructed* from ``nullif(1, 1)`` and typed. That construction lived
    only inside the SQL translator, which is why ``bt.lit(None)`` had no answer at all.

    Args:
        dtype: Arrow type name for the null, or ``"decimal(p,s)"``. Int64 when omitted.

    Returns:
        An expression that is NULL on every row, typed as `dtype`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir import null
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> ds.select(y=null("float64")).to_pydict()
            {'y': [None, None]}
    """
    dtype = _literal_dtype(dtype, "null()")
    one = Lit(1)
    untyped: Expr = NullIf(one, one)
    return untyped if dtype in (None, "int64") else untyped._cast(dtype, try_cast=False)


def null_of_type(dtype: pa.DataType) -> Expr | None:
    """A NULL of exactly `dtype`, or ``None`` when the cast vocabulary cannot name it.

    `null` takes a *name* from a short list; this takes the Arrow *type* a caller read off
    a schema, and names it through `dtype_name`, whose round trip is exact. So a column a
    rewrite has to fill with nulls (a diagonal concat's missing column, an INSERT's unlisted
    one) keeps its precision, scale, unit and time zone instead of being retyped to the
    nearest short name.

    Args:
        dtype: The Arrow type the null must have.

    Returns:
        An expression that is NULL on every row, typed as `dtype`, or ``None`` for a
        nested or extension type the cast grammar does not spell, and for the Arrow
        ``null`` type, which the engine cannot cast to.
    """
    name = dtype_name(dtype)
    if name is None or name == "null":
        return None
    one = Lit(1)
    return NullIf(one, one)._cast(name, try_cast=False)
