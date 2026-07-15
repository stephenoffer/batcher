"""The scalar expression base class and its core IR nodes.

`Expr` is the single expression representation in Batcher. The Python side builds
it (with operator overloading, so `col("x") > 2` is natural) and serializes it
via `to_ir()` to the exact JSON document the Rust `bc-expr` crate deserializes —
the same IR consumed by both the interpreter and (later) the JIT. The wire tags
here (`e`, `op`, literal kind) are a contract with the engine; keep them in sync.

This module holds the `Expr` base class plus the node classes that `Expr`'s own
methods construct. Leaf nodes that `Expr` does not build (`Col`, `Case`,
`CaseBuilder`, `NullIf`, `Greatest`, `Least`) live in
`batcher.plan.expr_ir.nodes`, and the accessor namespace classes and the nodes
they build live in `batcher.plan.expr_ir.namespaces`; the
`.str`/`.dt`/`.list`/`.struct`/`.json` properties import the latter lazily to
avoid an import-time cycle.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import ExprTag
from batcher.plan.types import CAST_DTYPES

if TYPE_CHECKING:
    from batcher.plan.expr_ir.audio import _AudioNamespace
    from batcher.plan.expr_ir.image import _ImageNamespace
    from batcher.plan.expr_ir.namespaces import (
        _DtNamespace,
        _JsonNamespace,
        _ListNamespace,
        _MapNamespace,
        _StrNamespace,
        _StructNamespace,
    )
    from batcher.plan.expr_ir.nodes import WindowExpr
    from batcher.plan.expr_ir.video import _VideoNamespace

# A value that can be promoted to an expression: another Expr or a Python scalar.
IntoExpr = Union["Expr", int, float, bool, str]


def _wrap(value: IntoExpr) -> Expr:
    return value if isinstance(value, Expr) else Lit(value)


def _col_or_expr(value: IntoExpr) -> Expr:
    """An ordering/source argument: a bare string names a *column*, not a string literal.

    ``_wrap`` would turn ``arg_max(v, "k")`` into an ordering by the constant ``'k'``;
    an ``Expr`` passes through unchanged. Mirrors SQL ``arg_max(v, k)`` / DuckDB.
    """
    from batcher.plan.expr_ir.constructors import col

    return col(value) if isinstance(value, str) else _wrap(value)


def _cut_labels(edges: list[float], left_closed: bool) -> list[str]:
    """Interval notation for `Expr.cut`'s bins, e.g. ``["(-inf, 1]", "(1, inf]"]``."""
    bounds = [float("-inf"), *edges, float("inf")]
    open_, close = ("[", ")") if left_closed else ("(", "]")
    return [
        f"{open_}{_cut_edge(lo)}, {_cut_edge(hi)}{close}" for lo, hi in itertools.pairwise(bounds)
    ]


def _cut_edge(value: float) -> str:
    """Render a bin edge: infinities by name, and whole floats without a `.0` tail."""
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return str(int(value)) if value.is_integer() else str(value)


class Expr:
    """Base class for scalar expressions — the one expression type in Batcher.

    An ``Expr`` is an immutable IR node, built lazily with operator overloading and
    fluent methods (``col("x") * 2``, ``col("x").sqrt()``, ``col("g").sum()``) and
    serialized via :meth:`to_ir` to the JSON the Rust ``bc-expr`` engine evaluates —
    no Python touches a row. Methods come in families: arithmetic/comparison/boolean
    operators, math functions (``sqrt``, ``ln``, ``sin``, …), null/NaN predicates
    (``is_null``, ``is_nan``, ``fill_null``, ``fill_nan``), aggregates for
    ``group_by().agg(...)`` / ``.over(...)`` (``sum``, ``mean``, ``count``, …), window
    helpers (``cum_sum``, ``shift``, ``diff``, ``pct_change``, ``rank``,
    ``rolling_mean``, ``is_unique``), and the typed accessor namespaces (``.str``,
    ``.dt``, ``.list``, ``.struct``, ``.json``, ``.image``, ``.audio``, ``.video``,
    ``.map``) that hold the per-type breadth.

    Subclasses are the concrete IR nodes (``Lit``, ``Binary``, ``MathExpr``, …); user
    code constructs expressions through ``col``/``lit`` and these methods, not the
    node classes directly.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2], "y": [10, 20]})
            >>> ds.select(z=bt.col("x") * bt.col("y") + 1).to_pydict()
            {'z': [11, 41]}
    """

    # --- serialization -----------------------------------------------------
    def to_ir(self) -> dict[str, Any]:  # pragma: no cover - overridden
        """Serialize this expression to its JSON IR dict — the wire contract with the engine.

        Each node emits ``{"e": <tag>, ...}`` matching the ``bc_expr::Expr`` serde
        tags the Rust interpreter and JIT deserialize. Overridden by every subclass;
        the base raises ``NotImplementedError``. Internal — not part of the user API.

        Returns:
            The node's JSON IR dict, tagged with its ``"e"`` wire kind.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("x").to_ir()
                {'e': 'col', 'name': 'x'}
        """
        raise NotImplementedError

    # --- comparison operators (yield boolean expressions) ------------------
    def __gt__(self, other: IntoExpr) -> Expr:
        """Element-wise greater-than (``a > b``), yielding a boolean expression."""
        return Binary("gt", self, _wrap(other))

    def __ge__(self, other: IntoExpr) -> Expr:
        """Element-wise greater-than-or-equal (``a >= b``), yielding a boolean expression."""
        return Binary("ge", self, _wrap(other))

    def __lt__(self, other: IntoExpr) -> Expr:
        """Element-wise less-than (``a < b``), yielding a boolean expression."""
        return Binary("lt", self, _wrap(other))

    def __le__(self, other: IntoExpr) -> Expr:
        """Element-wise less-than-or-equal (``a <= b``), yielding a boolean expression."""
        return Binary("le", self, _wrap(other))

    def __eq__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        """Element-wise equality (``a == b``), yielding a boolean expression (not a Python bool)."""
        return Binary("eq", self, _wrap(other))

    def __ne__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        """Element-wise inequality (``a != b``), yielding a boolean expression."""
        return Binary("ne", self, _wrap(other))

    # Expr is used for plan building, not as a dict key; make that explicit.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """A source-like rendering of the expression, e.g. ``(col('x') + lit(1))``."""
        from batcher.plan.expr_ir.render import render_expr

        return render_expr(self)

    # --- arithmetic operators ---------------------------------------------
    def __add__(self, other: IntoExpr) -> Expr:
        """Element-wise addition (``a + b``); also the string-concat operator on Utf8."""
        return Binary("add", self, _wrap(other))

    def __sub__(self, other: IntoExpr) -> Expr:
        """Element-wise subtraction (``a - b``)."""
        return Binary("sub", self, _wrap(other))

    def __mul__(self, other: IntoExpr) -> Expr:
        """Element-wise multiplication (``a * b``)."""
        return Binary("mul", self, _wrap(other))

    def __truediv__(self, other: IntoExpr) -> Expr:
        """Element-wise true division (``a / b``, → Float64); ``//`` is :meth:`__floordiv__`.

        The numerator is cast to Float64 so integer operands divide *truly*
        (``1 / 2`` is ``0.5``, as in Python, Polars and DuckDB) rather than
        truncating. Desugars to existing ops — no new IR — and the cast is free when
        the input is already Float64."""
        return Binary("div", self.cast("float64"), _wrap(other))

    def __mod__(self, other: IntoExpr) -> Expr:
        """Element-wise modulo / remainder (``a % b``)."""
        return Binary("mod", self, _wrap(other))

    # reflected forms so `2 * col("x")` works
    def __radd__(self, other: IntoExpr) -> Expr:
        """Reflected addition so ``scalar + expr`` works (also string concat on Utf8)."""
        return Binary("add", _wrap(other), self)

    def __rsub__(self, other: IntoExpr) -> Expr:
        """Reflected subtraction so ``scalar - expr`` works."""
        return Binary("sub", _wrap(other), self)

    def __rmul__(self, other: IntoExpr) -> Expr:
        """Reflected multiplication so ``scalar * expr`` works."""
        return Binary("mul", _wrap(other), self)

    def __rtruediv__(self, other: IntoExpr) -> Expr:
        """Reflected true division so ``scalar / expr`` works (→ Float64)."""
        return Binary("div", _wrap(other).cast("float64"), self)

    def __rmod__(self, other: IntoExpr) -> Expr:
        """Reflected modulo so ``scalar % expr`` works."""
        return Binary("mod", _wrap(other), self)

    def __floordiv__(self, other: IntoExpr) -> Expr:
        """Floor division ``a // b`` — ``floor(a / b)`` (Polars/Python semantics:
        rounds toward negative infinity, unlike SQL integer division which truncates
        toward zero). The numerator is cast to Float64 so the division is true
        (not integer) division before flooring; desugars to existing ops, no new IR."""
        return MathExpr("floor", Binary("div", self.cast("float64"), _wrap(other)))

    def __rfloordiv__(self, other: IntoExpr) -> Expr:
        """Reflected floor division so ``scalar // expr`` works; see :meth:`__floordiv__`."""
        return MathExpr("floor", Binary("div", _wrap(other).cast("float64"), self))

    # --- unary arithmetic operators ----------------------------------------
    def __neg__(self) -> Expr:
        """Arithmetic negation ``-x`` (desugars to ``0 - x``; type-preserving)."""
        return Binary("sub", Lit(0), self)

    def __pos__(self) -> Expr:
        """Unary plus ``+x`` — the identity, returning this expression unchanged."""
        return self

    def __abs__(self) -> MathExpr:
        """Absolute value ``abs(x)`` (Python ``abs()`` protocol)."""
        return MathExpr("abs", self)

    def __round__(self, ndigits: int | None = None) -> Expr:
        """Python ``round(expr)`` / ``round(expr, n)`` → :meth:`round`."""
        return self.round(ndigits)

    def __floor__(self) -> MathExpr:
        """``math.floor(expr)`` — round toward negative infinity."""
        return MathExpr("floor", self)

    def __ceil__(self) -> MathExpr:
        """``math.ceil(expr)`` — round toward positive infinity."""
        return MathExpr("ceil", self)

    def __trunc__(self) -> MathExpr:
        """``math.trunc(expr)`` — round toward zero."""
        return MathExpr("trunc", self)

    def __bool__(self) -> bool:
        """Guard against using an expression in a boolean context.

        ``col("x") > 0`` builds an expression; it has no truth value. Python would
        otherwise treat it as truthy in ``if expr:``, ``expr in (...)``, or
        ``a < expr < b`` (chained comparison) — silent logic bugs. Use ``&``/``|``/
        ``~`` to combine predicates and `is_in`/`between` for membership/ranges.
        """
        raise PlanError(
            "the truth value of an Expr is ambiguous; use & | ~ to combine predicates, "
            "and is_in()/between() instead of chained comparisons or `in`"
        )

    # --- boolean operators (bitwise spelling, like Polars/pandas) ----------
    def __and__(self, other: IntoExpr) -> Expr:
        """Boolean AND of two predicates (``a & b``), following SQL three-valued logic."""
        return Binary("and", self, _wrap(other))

    def __or__(self, other: IntoExpr) -> Expr:
        """Boolean OR of two predicates (``a | b``), following SQL three-valued logic."""
        return Binary("or", self, _wrap(other))

    # reflected forms so `True & col("x")` / `lit_on_left | col(...)` work
    def __rand__(self, other: IntoExpr) -> Expr:
        """Reflected boolean AND so ``scalar & expr`` works."""
        return Binary("and", _wrap(other), self)

    def __ror__(self, other: IntoExpr) -> Expr:
        """Reflected boolean OR so ``scalar | expr`` works."""
        return Binary("or", _wrap(other), self)

    def __invert__(self) -> Expr:
        """Boolean NOT of a predicate (``~a``), following SQL three-valued logic."""
        return Not(self)

    def __xor__(self, other: IntoExpr) -> Expr:
        """Bitwise XOR ``a ^ b`` of two integer expressions (operands cast to Int64);
        the operator spelling of :meth:`bitwise_xor`."""
        return Binary("bit_xor", self, _wrap(other))

    def __lshift__(self, other: IntoExpr) -> Expr:
        """Left shift ``a << b``; the operator spelling of :meth:`bitwise_left_shift`."""
        return Binary("shift_left", self, _wrap(other))

    def __rshift__(self, other: IntoExpr) -> Expr:
        """Right shift ``a >> b``; the operator spelling of :meth:`bitwise_right_shift`."""
        return Binary("shift_right", self, _wrap(other))

    def __rxor__(self, other: IntoExpr) -> Expr:
        """Reflected bitwise XOR so ``scalar ^ expr`` works (operands cast to Int64)."""
        return Binary("bit_xor", _wrap(other), self)

    def __rlshift__(self, other: IntoExpr) -> Expr:
        """Reflected left shift so ``scalar << expr`` works."""
        return Binary("shift_left", _wrap(other), self)

    def __rrshift__(self, other: IntoExpr) -> Expr:
        """Reflected right shift so ``scalar >> expr`` works."""
        return Binary("shift_right", _wrap(other), self)

    def __getitem__(self, key: int | slice | str) -> Expr:
        """Index into a list or struct column with ``[]`` (delegates to ``.list``/``.struct``).

        The idiomatic spelling of the accessors:

        - ``col("a")[2]`` → list element at index 2 (negative counts from the end),
          equivalent to ``col("a").list.get(2)``.
        - ``col("a")[1:3]`` → list sub-range ``[1, 3)``, equivalent to
          ``col("a").list.slice(1, 2)`` (a ``step`` other than 1 raises).
        - ``col("s")["field"]`` → struct field, equivalent to
          ``col("s").struct.field("field")``.

        Args:
            key: An int list index, a slice for a list sub-range, or a str struct
                field name.

        Returns:
            A new expression selecting the indexed element, sub-range, or field.

        Raises:
            PlanError: If `key` is a bool, has an unsupported type, or is a slice with
                a step other than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[10, 20, 30]]})
                >>> ds.select(r=bt.col("a")[1]).to_pydict()
                {'r': [20]}
        """
        from batcher.plan.expr_ir.func_nodes import ListGet, ListSlice, StructField

        if isinstance(key, bool):  # bool is an int subclass; reject it explicitly
            raise PlanError("cannot index an expression with a bool")
        if isinstance(key, int):
            return ListGet(self, key)
        if isinstance(key, str):
            return StructField(self, key)
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise PlanError("expression slice does not support a step other than 1")
            offset = key.start or 0
            length = None if key.stop is None else max(0, key.stop - offset)
            return ListSlice(self, offset, length)
        raise PlanError(f"cannot index an expression with {type(key).__name__}")

    def __iter__(self) -> Any:
        """Refuse iteration: an expression is a scalar column, not a sequence.

        `__getitem__` accepts an int index (``col("a")[2]`` → list element), which makes an
        expression *look* iterable to ``list(expr)`` / ``for x in expr`` — but the index has
        no upper bound (every ``expr[i]`` yields a fresh node), so the default iteration
        protocol would loop forever and exhaust memory. Raising here turns any accidental
        ``list(expr)`` (e.g. ``over(partition_by=col("g"))``) into an immediate, clear error.
        """
        raise TypeError(
            "a batcher expression is not iterable; wrap it in a list "
            "(e.g. over(partition_by=[col('g')]), not over(partition_by=col('g')))"
        )

    # --- bitwise integer operators (distinct from the boolean `&`/`|`) ------
    def bitwise_and(self, other: IntoExpr) -> Expr:
        """Bitwise AND ``self & other`` of two integer expressions.

        Operates per row on the integer bit patterns (operands cast to Int64), unlike
        the ``&`` operator which is boolean AND on predicates. The method spelling is
        unambiguous; nulls propagate.

        Args:
            other: The right-hand integer expression.

        Returns:
            A new integer expression of the bitwise AND.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [6], "b": [3]})
                >>> ds.select(r=bt.col("a").bitwise_and(bt.col("b"))).to_pydict()
                {'r': [2]}
        """
        return Binary("bit_and", self, _wrap(other))

    def bitwise_or(self, other: IntoExpr) -> Expr:
        """Bitwise OR ``self | other`` of two integers (per-row; Int64; nulls propagate).

        Args:
            other: The right-hand integer expression.

        Returns:
            A new integer expression of the bitwise OR.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [6], "b": [3]})
                >>> ds.select(r=bt.col("a").bitwise_or(bt.col("b"))).to_pydict()
                {'r': [7]}
        """
        return Binary("bit_or", self, _wrap(other))

    def bitwise_xor(self, other: IntoExpr) -> Expr:
        """Bitwise XOR ``self ^ other`` of two integers (per-row; Int64; nulls propagate).

        Args:
            other: The right-hand integer expression.

        Returns:
            A new integer expression of the bitwise XOR.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [6], "b": [3]})
                >>> ds.select(r=bt.col("a").bitwise_xor(bt.col("b"))).to_pydict()
                {'r': [5]}
        """
        return Binary("bit_xor", self, _wrap(other))

    def bitwise_left_shift(self, other: IntoExpr) -> Expr:
        """Left-shift this integer expression by `other` bits (per-row; Int64; nulls propagate).

        Args:
            other: The integer shift amount, in bits.

        Returns:
            A new integer expression of the left-shifted values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1]})
                >>> ds.select(r=bt.col("a").bitwise_left_shift(3)).to_pydict()
                {'r': [8]}
        """
        return Binary("shift_left", self, _wrap(other))

    def bitwise_right_shift(self, other: IntoExpr) -> Expr:
        """Right-shift this integer expression by `other` bits (per-row; Int64; nulls propagate).

        Args:
            other: The integer shift amount, in bits.

        Returns:
            A new integer expression of the right-shifted values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [16]})
                >>> ds.select(r=bt.col("a").bitwise_right_shift(2)).to_pydict()
                {'r': [4]}
        """
        return Binary("shift_right", self, _wrap(other))

    # --- naming ------------------------------------------------------------
    def alias(self, name: str) -> Aliased:
        """Bind an output name to this expression, for positional `select`.

        ``ds.select(col("a"), (col("x") * col("y")).alias("prod"))`` is equivalent
        to ``ds.select("a", prod=col("x") * col("y"))`` — `alias` just lets a
        derived column carry its name positionally. The alias is transparent in the
        IR (it serializes as the wrapped expression); only the projection layer
        reads it. `select`/`with_columns` keyword binding remains the canonical
        spelling — this is not a second way to project, only a positional name.

        Args:
            name: The output name to bind to this expression.

        Returns:
            The expression tagged with `name` for a positional `select`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.select((bt.col("x") * 2).alias("doubled")).to_pydict()
                {'doubled': [2, 4]}
        """
        return Aliased(self, name)

    # --- unary / type methods ----------------------------------------------
    def cast(self, dtype: str) -> Cast:
        """Cast to an Arrow type by name (int64/float64/int32/bool/string/...).

        The dtype is validated at plan-build time; an unknown name raises rather than
        failing opaquely in the engine mid-query. A value that cannot be converted
        errors the query (DuckDB ``CAST``); use `try_cast` to get NULL instead.

        Args:
            dtype: Target Arrow type name (e.g. ``"int64"``, ``"float64"``, ``"string"``).

        Returns:
            A new expression of the converted values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.select(r=bt.col("x").cast("float64")).to_pydict()
                {'r': [1.0, 2.0]}
        """
        return self._cast(dtype, try_cast=False)

    def try_cast(self, dtype: str) -> Cast:
        """Cast to an Arrow type by name; unconvertible values become NULL (DuckDB ``TRY_CAST``).

        The common safe-ingest spelling: ``col("x").try_cast("int64")`` turns a
        dirty string column into integers, with unparseable values becoming NULL
        (ready to `drop_nulls` or route to a quarantine sink).

        Args:
            dtype: Target Arrow type name (e.g. ``"int64"``, ``"float64"``, ``"string"``).

        Returns:
            A new expression of the converted values, NULL where conversion fails.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": ["1", "bad"]})
                >>> ds.select(r=bt.col("x").try_cast("int64")).to_pydict()
                {'r': [1, None]}
        """
        return self._cast(dtype, try_cast=True)

    def _cast(self, dtype: str, *, try_cast: bool) -> Cast:
        if dtype not in CAST_DTYPES:
            import difflib

            hint = difflib.get_close_matches(dtype, sorted(CAST_DTYPES), n=2, cutoff=0.5)
            suffix = f"; did you mean {' or '.join(map(repr, hint))}?" if hint else ""
            raise PlanError(f"unknown cast dtype {dtype!r}; valid: {sorted(CAST_DTYPES)}{suffix}")
        return Cast(self, dtype, try_cast=try_cast)

    def is_null(self) -> IsNull:
        """True where the value is NULL (SQL ``IS NULL``).

        A boolean expression that never itself yields null — a null input maps to
        true. Distinct from :meth:`is_nan`, which is the float-only NaN notion.

        Returns:
            A boolean expression, true where the value is null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3]})
                >>> ds.select(r=bt.col("x").is_null()).to_pydict()
                {'r': [False, True, False]}
        """
        return IsNull(self)

    def is_not_null(self) -> IsNotNull:
        """True where the value is non-NULL (SQL ``IS NOT NULL``); negation of :meth:`is_null`.

        Returns:
            A boolean expression, true where the value is non-null.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3]})
                >>> ds.select(r=bt.col("x").is_not_null()).to_pydict()
                {'r': [True, False, True]}
        """
        return IsNotNull(self)

    def is_in(self, values: Iterable[IntoExpr]) -> Expr:
        """``self IN (values)`` — true if equal to any value.

        Desugars to an OR of equality checks, so it follows SQL three-valued
        logic (``NULL IN (...)`` is NULL) and an empty collection is always false.

        Args:
            values: The scalars or expressions to test membership against.

        Returns:
            A boolean expression, true where the value is in `values`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.select(r=bt.col("x").is_in([1, 3])).to_pydict()
                {'r': [True, False, True]}
        """
        vals = list(values)
        # SQL three-valued logic: a NULL member never yields True, but it turns a
        # would-be False into NULL (``x IN (1, NULL)`` is True for x=1, NULL otherwise;
        # DuckDB agrees). A NULL member contributes an always-null disjunct, which
        # `nullif(lit(True), lit(True))` builds without a first-class null literal.
        has_null = any(v is None for v in vals)
        non_null = [v for v in vals if v is not None]
        if not non_null:
            if has_null:
                from batcher.plan.expr_ir.constructors import lit, nullif

                return nullif(lit(True), lit(True))
            return Lit(False)
        expr: Expr = self == non_null[0]
        for v in non_null[1:]:
            expr = expr | (self == v)
        if has_null:
            from batcher.plan.expr_ir.constructors import lit, nullif

            expr = expr | nullif(lit(True), lit(True))
        return expr

    def between(self, low: IntoExpr, high: IntoExpr, closed: str = "both") -> Expr:
        """``self BETWEEN low AND high``, matching SQL/DuckDB (both bounds inclusive by default).

        Desugars to a pair of comparisons, so it follows SQL three-valued logic — a
        null operand makes the result null. The idiomatic spelling for a range filter
        (chained comparisons like ``low <= col("x") <= high`` are rejected; see
        :meth:`__bool__`). Pass `closed` to make either bound exclusive (Polars
        ``is_between`` parity).

        Args:
            low: Lower bound.
            high: Upper bound.
            closed: Which bounds are inclusive — ``"both"`` (default), ``"left"``
                (``[low, high)``), ``"right"`` (``(low, high]``), or ``"none"``
                (``(low, high)``).

        Returns:
            A boolean expression, true where the value lies in the range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 10]})
                >>> ds.select(r=bt.col("x").between(2, 8)).to_pydict()
                {'r': [False, True, False]}

                >>> ds.select(r=bt.col("x").between(1, 10, closed="none")).to_pydict()
                {'r': [False, True, False]}
        """
        if closed not in ("both", "left", "right", "none"):
            raise PlanError(
                f"between(closed=...) must be 'both', 'left', 'right', or 'none', got {closed!r}"
            )
        lo, hi = _wrap(low), _wrap(high)
        lower = self >= lo if closed in ("both", "left") else self > lo
        upper = self <= hi if closed in ("both", "right") else self < hi
        return lower & upper

    def eq_missing(self, other: IntoExpr) -> Expr:
        """Null-safe equality (SQL ``IS NOT DISTINCT FROM``) where two nulls compare equal.

        A null compared with a non-null is **false** (never null). The reliable way
        to compare possibly-null keys — used for change detection
        in slowly-changing dimensions. Desugars to existing ops (no new IR).

        Args:
            other: The expression or scalar to compare against.

        Returns:
            A boolean expression of the null-safe comparison.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, None], "b": [1, None]})
                >>> ds.select(r=bt.col("a").eq_missing(bt.col("b"))).to_pydict()
                {'r': [True, True]}
        """
        o = _wrap(other)
        both_null = self.is_null() & o.is_null()
        return Coalesce([self == o, Lit(False)]) | both_null

    def replace(self, mapping: dict[Any, Any], *, default: IntoExpr | None = None) -> Expr:
        """Remap values through a ``{old: new}`` dictionary (a value standardization / recode).

        Values absent from `mapping` keep their original value, or take `default`
        when one is given. Desugars to a ``CASE`` chain (no new IR).

        ``col("c").replace({"US": "USA", "UK": "GBR"})`` standardizes country codes.

        Args:
            mapping: A ``{old: new}`` dict of replacements.
            default: Value for entries absent from `mapping`; ``None`` keeps the original.

        Returns:
            A new expression with mapped values substituted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["US", "UK", "FR"]})
                >>> ds.select(r=bt.col("c").replace({"US": "USA", "UK": "GBR"})).to_pydict()
                {'r': ['USA', 'GBR', 'FR']}
        """
        from batcher.plan.expr_ir.constructors import when

        if not mapping:
            return self if default is None else _wrap(default)
        items = list(mapping.items())
        builder = when(self == _wrap(items[0][0])).then(_wrap(items[0][1]))
        for old, new in items[1:]:
            builder = builder.when(self == _wrap(old)).then(_wrap(new))
        return builder.otherwise(self if default is None else _wrap(default))

    @property
    def str(self) -> _StrNamespace:
        """String-function accessor — grouped string ops on this (string) column.

        Returns a namespace holding string transforms and predicates such as
        ``.str.upper()``, ``.str.contains("x")``, ``.str.replace(...)``,
        ``.str.slice(...)``, and ``.str.len()``.

        Returns:
            The `.str` string-function accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": ["ab", "cd"]})
                >>> ds.select(r=bt.col("s").str.upper()).to_pydict()
                {'r': ['AB', 'CD']}
        """
        from batcher.plan.expr_ir.namespaces import _StrNamespace

        return _StrNamespace(self)

    @property
    def dt(self) -> _DtNamespace:
        """Date/time accessor — grouped temporal field extraction on this (date/timestamp) column.

        Returns a namespace with components such as ``.dt.year()``, ``.dt.month()``,
        ``.dt.day()``, ``.dt.hour()``, and ``.dt.weekday()``.

        Returns:
            The `.dt` date/time accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime
                >>> ds = bt.from_pydict({"d": [datetime.date(2021, 5, 3)]})
                >>> ds.select(r=bt.col("d").dt.year()).to_pydict()
                {'r': [2021]}
        """
        from batcher.plan.expr_ir.namespaces import _DtNamespace

        return _DtNamespace(self)

    # --- math functions ----------------------------------------------------
    def abs(self) -> MathExpr:
        """Absolute value, preserving the input numeric dtype (nulls propagate).

        Returns:
            A new expression of the absolute values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, -2, 3]})
                >>> ds.select(r=bt.col("x").abs()).to_pydict()
                {'r': [1, 2, 3]}
        """
        return MathExpr("abs", self)

    def round(self, digits: int | None = None) -> Expr:
        """Round half-away-from-zero to the nearest integer, or to `digits` decimal places.

        Args:
            digits: Number of decimal places to keep. ``None`` (the default) rounds to
                a whole number.

        Returns:
            A new expression of the rounded values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.234, 2.567]})
                >>> ds.select(r=bt.col("x").round(2)).to_pydict()
                {'r': [1.23, 2.57]}
        """
        if digits is None:
            return MathExpr("round", self)
        return Math2Expr("round", self, Lit(digits))

    def pow(self, exponent: IntoExpr) -> Math2Expr:
        """This value raised to `exponent` (→ Float64); the method spelling of the ``**`` operator.

        Args:
            exponent: A scalar or expression power; applied per row, nulls propagate.

        Returns:
            A new Float64 expression of the powers.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [2.0, 3.0]})
                >>> ds.select(r=bt.col("x").pow(2)).to_pydict()
                {'r': [4.0, 9.0]}
        """
        return Math2Expr("pow", self, _wrap(exponent))

    def __pow__(self, other: IntoExpr) -> Math2Expr:
        """Exponentiation (``a ** b``, → Float64); the operator spelling of :meth:`pow`."""
        return Math2Expr("pow", self, _wrap(other))

    def __rpow__(self, other: IntoExpr) -> Math2Expr:
        """Reflected exponentiation so ``scalar ** expr`` works (→ Float64)."""
        return Math2Expr("pow", _wrap(other), self)

    def floor(self) -> MathExpr:
        """Round down toward negative infinity to the nearest integer value (nulls propagate).

        Returns:
            A new expression rounded toward negative infinity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.2, 2.8]})
                >>> ds.select(r=bt.col("x").floor()).to_pydict()
                {'r': [1.0, 2.0]}
        """
        return MathExpr("floor", self)

    def ceil(self) -> MathExpr:
        """Round up toward positive infinity to the nearest integer value (nulls propagate).

        Returns:
            A new expression rounded toward positive infinity.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.2, 2.8]})
                >>> ds.select(r=bt.col("x").ceil()).to_pydict()
                {'r': [2.0, 3.0]}
        """
        return MathExpr("ceil", self)

    def sqrt(self) -> MathExpr:
        """Square root (→ Float64). Negative inputs yield NaN; nulls propagate.

        Returns:
            A new Float64 expression of the square roots.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [4.0, 9.0]})
                >>> ds.select(r=bt.col("x").sqrt()).to_pydict()
                {'r': [2.0, 3.0]}
        """
        return MathExpr("sqrt", self)

    def ln(self) -> MathExpr:
        """Natural logarithm, base e (→ Float64). Non-positive inputs yield NaN/-inf; nulls keep.

        Returns:
            A new Float64 expression of the natural logarithms.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import math
                >>> ds = bt.from_pydict({"x": [1.0, math.e]})
                >>> ds.select(r=bt.col("x").ln()).to_pydict()
                {'r': [0.0, 1.0]}
        """
        return MathExpr("ln", self)

    def log10(self) -> MathExpr:
        """Base-10 logarithm (→ Float64). Non-positive inputs yield NaN/-inf; nulls propagate.

        Returns:
            A new Float64 expression of the base-10 logarithms.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 100.0]})
                >>> ds.select(r=bt.col("x").log10()).to_pydict()
                {'r': [0.0, 2.0]}
        """
        return MathExpr("log10", self)

    def log2(self) -> MathExpr:
        """Base-2 logarithm (→ Float64). Non-positive inputs yield NaN/-inf; nulls propagate.

        Returns:
            A new Float64 expression of the base-2 logarithms.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 8.0]})
                >>> ds.select(r=bt.col("x").log2()).to_pydict()
                {'r': [0.0, 3.0]}
        """
        return MathExpr("log2", self)

    def exp(self) -> MathExpr:
        """``e`` raised to this value, the inverse of :meth:`ln` (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of ``e`` raised to each value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").exp()).to_pydict()
                {'r': [1.0, 2.718281828459045]}
        """
        return MathExpr("exp", self)

    def sin(self) -> MathExpr:
        """Sine of an angle given in radians (→ Float64; nulls propagate). See :meth:`radians`.

        Returns:
            A new Float64 expression of the sines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").sin()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("sin", self)

    def cos(self) -> MathExpr:
        """Cosine of an angle given in radians (→ Float64; nulls propagate). See :meth:`radians`.

        Returns:
            A new Float64 expression of the cosines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").cos()).to_pydict()
                {'r': [1.0]}
        """
        return MathExpr("cos", self)

    def tan(self) -> MathExpr:
        """Tangent of an angle in radians (→ Float64; nulls propagate). See :meth:`radians`.

        Returns:
            A new Float64 expression of the tangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").tan()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("tan", self)

    def sign(self) -> MathExpr:
        """Sign of the value as ``-1.0``, ``0.0``, or ``1.0`` (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the signs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-5.0, 0.0, 5.0]})
                >>> ds.select(r=bt.col("x").sign()).to_pydict()
                {'r': [-1.0, 0.0, 1.0]}
        """
        return MathExpr("sign", self)

    def trunc(self) -> MathExpr:
        """Truncate toward zero, dropping the fractional part (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the truncated values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.7, -1.7]})
                >>> ds.select(r=bt.col("x").trunc()).to_pydict()
                {'r': [1.0, -1.0]}
        """
        return MathExpr("trunc", self)

    def cbrt(self) -> MathExpr:
        """Cube root (→ Float64; defined for negatives, unlike :meth:`sqrt`; nulls propagate).

        Returns:
            A new Float64 expression of the cube roots.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [8.0, 27.0]})
                >>> ds.select(r=bt.col("x").cbrt()).to_pydict()
                {'r': [2.0, 3.0]}
        """
        return MathExpr("cbrt", self)

    def asin(self) -> MathExpr:
        """Arcsine in radians, inverse of :meth:`sin` (→ Float64; outside [-1, 1] → NaN).

        Returns:
            A new Float64 expression of the arcsines, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").asin()).to_pydict()
                {'r': [0.0, 1.5707963267948966]}
        """
        return MathExpr("asin", self)

    def acos(self) -> MathExpr:
        """Arccosine in radians, inverse of :meth:`cos` (→ Float64; outside [-1, 1] → NaN).

        Returns:
            A new Float64 expression of the arccosines, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0]})
                >>> ds.select(r=bt.col("x").acos()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("acos", self)

    def atan(self) -> MathExpr:
        """Arctangent in radians, the inverse of :meth:`tan` (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the arctangents, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").atan()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("atan", self)

    def sinh(self) -> MathExpr:
        """Hyperbolic sine (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the hyperbolic sines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").sinh()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("sinh", self)

    def cosh(self) -> MathExpr:
        """Hyperbolic cosine (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the hyperbolic cosines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").cosh()).to_pydict()
                {'r': [1.0]}
        """
        return MathExpr("cosh", self)

    def tanh(self) -> MathExpr:
        """Hyperbolic tangent (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the hyperbolic tangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").tanh()).to_pydict()
                {'r': [0.0]}
        """
        return MathExpr("tanh", self)

    def degrees(self) -> MathExpr:
        """Convert an angle from radians to degrees (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the angles in degrees.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import math
                >>> ds = bt.from_pydict({"x": [math.pi]})
                >>> ds.select(r=bt.col("x").degrees()).to_pydict()
                {'r': [180.0]}
        """
        return MathExpr("degrees", self)

    def radians(self) -> MathExpr:
        """Convert an angle from degrees to radians (→ Float64; nulls propagate).

        The trig functions (:meth:`sin`/:meth:`cos`/:meth:`tan`) expect radians, so
        pair this with them when starting from degrees.

        Returns:
            A new Float64 expression of the angles in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [180.0]})
                >>> ds.select(r=bt.col("x").radians()).to_pydict()
                {'r': [3.141592653589793]}
        """
        return MathExpr("radians", self)

    def cot(self) -> MathExpr:
        """Cotangent (``1 / tan``) of an angle in radians (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the cotangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0]})
                >>> ds.select(r=bt.col("x").cot()).to_pydict()
                {'r': [0.6420926159343306]}
        """
        return MathExpr("cot", self)

    def factorial(self) -> MathExpr:
        """``n!`` — factorial of a non-negative integer (DuckDB ``factorial``; → Float64).

        Returns:
            A new Float64 expression of the factorials.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5]})
                >>> ds.select(f=bt.col("x").factorial()).to_pydict()
                {'f': [120.0]}
        """
        return MathExpr("factorial", self)

    def bit_count(self) -> MathExpr:
        """Population count — the number of set bits in the integer value (DuckDB ``bit_count``).

        Returns:
            A new expression of the set-bit counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [7]})
                >>> ds.select(r=bt.col("x").bit_count()).to_pydict()
                {'r': [3.0]}
        """
        return MathExpr("bit_count", self)

    @property
    def list(self) -> _ListNamespace:
        """List accessor — grouped per-row reductions and element access on a list column.

        Returns a namespace with ops such as ``.list.len()``, ``.list.sum()``,
        ``.list.get(i)``, ``.list.slice(offset, length)``, and ``.list.join(sep)``.

        Returns:
            The `.list` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [[1, 2], [3]]})
                >>> ds.select(r=bt.col("a").list.len()).to_pydict()
                {'r': [2, 1]}
        """
        from batcher.plan.expr_ir.namespaces import _ListNamespace

        return _ListNamespace(self)

    @property
    def struct(self) -> _StructNamespace:
        """Struct accessor — grouped field access on a struct column, e.g. ``.struct.field("x")``.

        Returns a namespace whose ``.field(name)`` projects a named field as a column.

        Returns:
            The `.struct` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"a": 1}, {"a": 2}]})
                >>> ds.select(r=bt.col("s").struct.field("a")).to_pydict()
                {'r': [1, 2]}
        """
        from batcher.plan.expr_ir.namespaces import _StructNamespace

        return _StructNamespace(self)

    @property
    def map(self) -> _MapNamespace:
        """Map accessor — grouped key/value access on a map column.

        Returns a namespace with ``.map.keys()``, ``.map.values()``, and
        ``.map.get(key)``.

        Returns:
            The `.map` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("m").map.get("k").to_ir()["e"]
                'map'
        """
        from batcher.plan.expr_ir.namespaces import _MapNamespace

        return _MapNamespace(self)

    @property
    def json(self) -> _JsonNamespace:
        """JSON accessor — grouped JSONPath extraction on a JSON-string column.

        Returns a namespace with typed extractors such as
        ``.json.extract_string("$.a")``, evaluated in the engine (no Python parsing).

        Returns:
            The `.json` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"j": ['{"a": "x"}']})
                >>> ds.select(r=bt.col("j").json.extract_string("$.a")).to_pydict()
                {'r': ['x']}
        """
        from batcher.plan.expr_ir.namespaces import _JsonNamespace

        return _JsonNamespace(self)

    @property
    def image(self) -> _ImageNamespace:
        """Image accessor — grouped lazy image-decode ops on a binary column.

        Returns a namespace with ops such as ``.image.decode()`` and
        ``.image.to_tensor(224, 224)``; decoding stays in the Rust data plane.

        Returns:
            The `.image` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> Expr = type(bt.col("x")).__mro__[1]
                >>> isinstance(bt.col("img").image.decode(), Expr)
                True
        """
        from batcher.plan.expr_ir.image import _ImageNamespace

        return _ImageNamespace(self)

    @property
    def audio(self) -> _AudioNamespace:
        """Audio accessor — grouped lazy audio-decode ops on a binary column.

        Returns a namespace with ops such as ``.audio.decode()`` and
        ``.audio.to_waveform()``.

        Returns:
            The `.audio` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("a").audio.decode().to_ir()["e"]
                'audio'
        """
        from batcher.plan.expr_ir.audio import _AudioNamespace

        return _AudioNamespace(self)

    @property
    def video(self) -> _VideoNamespace:
        """Video accessor — grouped lazy video-decode ops on a binary column.

        Returns a namespace with ops such as ``.video.decode()`` (requires the engine
        built with the ``video`` feature).

        Returns:
            The `.video` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("v").video.decode().to_ir()["e"]
                'video'
        """
        from batcher.plan.expr_ir.video import _VideoNamespace

        return _VideoNamespace(self)

    def hash(self, seed: int = 0) -> Expr:
        """A deterministic 64-bit hash of this expression's value, per row → Int64.

        The single-argument spelling of :func:`batcher.hash_rows`. Typed rather than
        textual, so it neither depends on how a float renders nor pays to render it.

        Args:
            seed: Changes the digest; the same seed reproduces it.

        Returns:
            An Int64 expression — the value's digest.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 1, 2]})
                >>> h = ds.select(h=bt.col("x").hash()).to_pydict()["h"]
                >>> h[0] == h[1], h[0] == h[2]
                (True, False)
        """
        from batcher.plan.expr_ir.constructors import hash_rows

        return hash_rows(self, seed=seed)

    def fill_null(self, value: IntoExpr) -> Coalesce:
        """Replace nulls with `value`, leaving non-null values unchanged (SQL ``COALESCE``).

        `value` may be a scalar or another expression (e.g. a column to fall back to).
        Only NULL is replaced — float NaN is not a null, so use :meth:`is_nan` to
        handle it.

        Args:
            value: The replacement used wherever this expression is null.

        Returns:
            A new expression with every null replaced by `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3]})
                >>> ds.select(r=bt.col("x").fill_null(0)).to_pydict()
                {'r': [1, 0, 3]}
        """
        return Coalesce([self, _wrap(value)])

    # --- NaN handling / clamping -------------------------------------------
    def is_nan(self) -> Expr:
        """True where the value is IEEE NaN (a float-only notion, distinct from null).

        Nulls propagate (a null input yields null, not true). This is a dedicated op,
        not the ``self != self`` trick: the engine's ``!=`` uses total ordering
        (where ``NaN == NaN``), so ``self != self`` would never flag a NaN.

        Returns:
            A boolean expression, true where the value is NaN.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("nan"), 3.0]})
                >>> ds.select(r=bt.col("x").is_nan()).to_pydict()
                {'r': [False, True, False]}
        """
        return IsNan(self)

    def is_not_nan(self) -> Expr:
        """True where the float value is not IEEE NaN — the negation of :meth:`is_nan`.

        Nulls propagate (a null input yields null, not true). NaN is distinct from
        NULL; use :meth:`is_not_null` for the null check.

        Returns:
            A boolean expression, true where the value is not NaN.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("nan")]})
                >>> ds.select(r=bt.col("x").is_not_nan()).to_pydict()
                {'r': [True, False]}
        """
        return Not(IsNan(self))

    def is_infinite(self) -> Expr:
        """True where the value is ``+inf`` or ``-inf`` (Polars/pandas ``is_infinite``).

        A dedicated op because ``±inf`` literals do not survive the JSON IR, so a
        comparison against them cannot express this. Nulls propagate (null → null).

        Returns:
            A boolean expression, true where the value is infinite.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("inf")]})
                >>> ds.select(r=bt.col("x").is_infinite()).to_pydict()
                {'r': [False, True]}
        """
        return IsInf(self)

    def is_finite(self) -> Expr:
        """True where the value is finite — not NaN and not ``±inf`` (``is_finite``).

        Nulls propagate (null → null).

        Returns:
            A boolean expression, true where the value is finite.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("inf")]})
                >>> ds.select(r=bt.col("x").is_finite()).to_pydict()
                {'r': [True, False]}
        """
        return Not(IsNan(self)) & Not(IsInf(self))

    def clip(self, lower: IntoExpr | None = None, upper: IntoExpr | None = None) -> Expr:
        """Clamp values into ``[lower, upper]`` (either bound optional).

        Nulls are preserved (a null stays null, not pulled to a bound): the lowering
        is a conditional, so a comparison against a null input is null and falls
        through to the original value. NaN is likewise left untouched (matching
        Polars/pandas), even though the engine's total order ranks NaN above every
        finite value — an explicit guard re-injects it after the bounds are applied.

        Args:
            lower: Lower bound; ``None`` leaves the low side unclamped.
            upper: Upper bound; ``None`` leaves the high side unclamped.

        Returns:
            A new expression with the values clamped into the bounds.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 10]})
                >>> ds.select(r=bt.col("x").clip(2, 8)).to_pydict()
                {'r': [2, 5, 8]}
        """
        from batcher.plan.expr_ir.constructors import when

        result: Expr = self
        clamped = False
        if lower is not None:
            result = when(result < _wrap(lower)).then(lower).otherwise(result)
            clamped = True
        if upper is not None:
            result = when(result > _wrap(upper)).then(upper).otherwise(result)
            clamped = True
        if clamped:
            # NaN is total-order-greatest, so an upper bound would otherwise pull it
            # down to `upper`; Polars/pandas leave NaN alone. Restore the original.
            result = when(self.is_nan()).then(self).otherwise(result)
        return result

    # --- aggregate constructors (used inside group_by().agg(...)) -----------
    def sum(self) -> AggExpr:
        """Sum of non-null values per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        An aggregate: it collapses a group to one row (or, via :meth:`AggExpr.over`,
        broadcasts the group result to each row). Mergeable, so identical single-node
        and distributed.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(total=bt.col("x").sum()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'total': [3, 10]}
        """
        return AggExpr("sum", self)

    def min(self) -> AggExpr:
        """Minimum non-null value per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").min()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("min", self)

    def max(self) -> AggExpr:
        """Maximum non-null value per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").max()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        return AggExpr("max", self)

    def mean(self) -> AggExpr:
        """Arithmetic mean of non-null values per group (→ Float64).

        An aggregate for ``group_by().agg(...)`` / ``.over(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").mean()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1.5, 10.0]}
        """
        return AggExpr("mean", self)

    def var(self) -> AggExpr:
        """Sample variance per group, Bessel-corrected (divides by ``n - 1``).

        An aggregate for ``group_by().agg(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 4, 6]})
                >>> ds.group_by("g").agg(r=bt.col("x").var()).to_pydict()
                {'g': ['a'], 'r': [4.0]}
        """
        return AggExpr("var", self)

    def std(self) -> AggExpr:
        """Sample standard deviation per group — the square root of :meth:`var`.

        An aggregate for ``group_by().agg(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 4, 6]})
                >>> ds.group_by("g").agg(r=bt.col("x").std()).to_pydict()
                {'g': ['a'], 'r': [2.0]}
        """
        return AggExpr("stddev", self)

    def skewness(self) -> AggExpr:
        """Sample skewness per group (adjusted Fisher-Pearson, matching DuckDB; → Float64).

        Null when the group has fewer than 3 values. Mergeable (sum-of-powers moment state).

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").skewness()).to_pydict()
                {'g': ['a'], 'r': [1.763632614803888]}
        """
        return AggExpr("skewness", self)

    def kurtosis(self) -> AggExpr:
        """Sample excess kurtosis per group (0 for a normal distribution; → Float64).

        Matches DuckDB. Null when the group has fewer than 4 values. Mergeable.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").kurtosis()).to_pydict()
                {'g': ['a'], 'r': [3.152000000000008]}
        """
        return AggExpr("kurtosis", self)

    def median(self) -> AggExpr:
        """Exact median per group — the 0.5 quantile (→ Float64).

        Averages the two middle values for an even count. Equals ``quantile(0.5)``. An
        aggregate for ``group_by().agg(...)``; see :meth:`approx_median` for a
        bounded-memory sketch.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").median()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1.5, 10.0]}
        """
        return AggExpr("median", self)

    def quantile(self, q: float) -> AggExpr:
        """Continuous quantile at ``q`` in [0, 1] (linear interpolation).

        ``quantile(0.5)`` equals :meth:`median`. Raises ``PlanError`` if ``q`` is
        outside [0, 1].

        Args:
            q: The quantile in ``[0, 1]``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Raises:
            PlanError: If `q` is outside ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").quantile(0.5)).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1.5, 10.0]}
        """
        from batcher._internal.errors import PlanError

        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile q must be in [0, 1], got {q}")
        return AggExpr("quantile", self, param=float(q))

    def count(self) -> AggExpr:
        """Number of non-null values per group (SQL ``COUNT(expr)``; nulls are skipped).

        An aggregate for ``group_by().agg(...)``. For a row count that includes nulls,
        count a non-null key or use the top-level ``count()``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").count()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 1]}
        """
        return AggExpr("count", self)

    def n_unique(self) -> AggExpr:
        """Number of distinct non-null values per group (SQL ``COUNT(DISTINCT)``).

        Exact, so it holds every distinct value — see :meth:`approx_n_unique` for the
        bounded-memory, skew-safe sketch. An aggregate for ``group_by().agg(...)``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").n_unique()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 1]}
        """
        return AggExpr("count_distinct", self)

    # SQL spelling; same aggregate as `n_unique`.
    count_distinct = n_unique

    def approx_n_unique(self) -> AggExpr:
        """Approximate COUNT(DISTINCT) via a HyperLogLog sketch (~2% error).

        Bounded memory regardless of skew — the skew-safe choice when an exact
        `n_unique` on a hot key would hold every distinct value. Mergeable, so it
        is identical single-node and distributed.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [1, 2, 3]})
                >>> ds.group_by("g").agg(n=bt.col("x").approx_n_unique()).to_pydict()
                {'g': ['a'], 'n': [3]}
        """
        return AggExpr("approx_count_distinct", self)

    # SQL spelling; same aggregate as `approx_n_unique`.
    approx_count_distinct = approx_n_unique

    def approx_quantile(self, q: float) -> AggExpr:
        """Approximate quantile `q ∈ [0, 1]` via a KLL sketch (bounded memory).

        The skew-safe choice when an exact `quantile`/`median` on a hot key would
        hold every value. Mergeable, so identical single-node and distributed.

        Args:
            q: The quantile in ``[0, 1]``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Raises:
            PlanError: If `q` is outside ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "b"], "x": [10.0, 20.0]})
                >>> r = ds.group_by("g").agg(q=bt.col("x").approx_quantile(0.5)).sort("g")
                >>> r.with_columns(q=bt.col("q").round()).to_pydict()
                {'g': ['a', 'b'], 'q': [10.0, 20.0]}
        """
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"approx_quantile(q) requires q in [0, 1], got {q}")
        return AggExpr("approx_quantile", self, param=float(q))

    def approx_median(self) -> AggExpr:
        """Approximate median (the 0.5 quantile) via a KLL sketch — see :meth:`approx_quantile`.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "b"], "x": [10.0, 20.0]})
                >>> r = ds.group_by("g").agg(m=bt.col("x").approx_median()).sort("g")
                >>> r.with_columns(m=bt.col("m").round()).to_pydict()
                {'g': ['a', 'b'], 'm': [10.0, 20.0]}
        """
        return AggExpr("approx_quantile", self, param=0.5)

    def mode(self) -> AggExpr:
        """Most frequent value per group, ties broken by the smallest value.

        Deterministic and partition-independent. Works on any column type.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").mode()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("mode", self)

    def first(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the first row in `order_by` order (SQL ``first``).

        Equivalent to ``arg_min(order_by)``.

        An explicit `order_by` is **required**: an arrival-order first/last is not
        partition-independent, so it could not stay identical single-node and
        distributed. With an order key the result is deterministic and mergeable
        (ties on the key break to the smallest value).

        Args:
            order_by: The ordering expression; the value at its first row is returned.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").first(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        return AggExpr("arg_min", self, input2=_col_or_expr(order_by))

    def last(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the last row in `order_by` order (SQL ``last``).

        Equivalent to ``arg_max(order_by)``. As with :meth:`first`, an explicit
        `order_by` is required so the result stays deterministic and mergeable across
        partitions.

        Args:
            order_by: The ordering expression; the value at its last row is returned.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").last(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("arg_max", self, input2=_col_or_expr(order_by))

    def arg_min(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is minimal (SQL ``arg_min``/``min_by``).

        Key ties break to the smallest value, so the result is deterministic and
        partition-independent.

        Args:
            by: The expression whose minimum selects the row.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").arg_min(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        return AggExpr("arg_min", self, input2=_col_or_expr(by))

    def arg_max(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is maximal (SQL ``arg_max``/``max_by``).

        Args:
            by: The expression whose maximum selects the row.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").arg_max(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("arg_max", self, input2=_col_or_expr(by))

    def bool_and(self) -> AggExpr:
        """Logical AND of this boolean expression's non-null values per group.

        Null when the group has no non-null value.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, True]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_and()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [False, True]}
        """
        return AggExpr("bool_and", self)

    def bool_or(self) -> AggExpr:
        """Logical OR of this boolean expression's non-null values per group.

        Null when the group has no non-null value.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, False]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_or()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [True, False]}
        """
        return AggExpr("bool_or", self)

    def product(self) -> AggExpr:
        """Product of non-null values per group (DuckDB ``product``; → Float64).

        Mergeable, so identical single-node and distributed.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").product()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [6.0, 5.0]}
        """
        return AggExpr("product", self)

    def bit_and(self) -> AggExpr:
        """Bitwise AND of non-null Int64 values per group (Spark/DuckDB ``bit_and``).

        Mergeable.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_and()).to_pydict()
                {'g': ['a'], 'r': [2]}
        """
        return AggExpr("bit_and", self)

    def bit_or(self) -> AggExpr:
        """Bitwise OR of non-null Int64 values per group (Spark/DuckDB ``bit_or``).

        Mergeable.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_or()).to_pydict()
                {'g': ['a'], 'r': [7]}
        """
        return AggExpr("bit_or", self)

    def bit_xor(self) -> AggExpr:
        """Bitwise XOR of non-null Int64 values per group (Spark/DuckDB ``bit_xor``).

        Mergeable.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_xor()).to_pydict()
                {'g': ['a'], 'r': [5]}
        """
        return AggExpr("bit_xor", self)

    def histogram(self) -> AggExpr:
        """Collect non-null values per group into a ``Map<value, count>`` (DuckDB ``histogram``).

        Keys are the distinct values sorted ascending; values are their counts.
        Mergeable, so identical single-node and distributed.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 2]})
                >>> ds.group_by("g").agg(r=bt.col("x").histogram()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [[(1, 2)], [(2, 1)]]}
        """
        return AggExpr("histogram", self)

    def array_agg(self) -> AggExpr:
        """Collect non-null values in each group into a ``List`` (SQL ``array_agg``).

        Spark ``collect_list``. Without an explicit order the element order is
        arrival-dependent. Mergeable — the per-group value list is the partial state,
        so the result is the same single-node and distributed.

        Chain a list reduction on the result column to summarize it, e.g.
        ``ds.group_by("g").agg(tags=col("t").array_agg())`` then
        ``col("tags").list.join(",")``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").array_agg()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [[1, 2], [10]]}
        """
        return AggExpr("list_agg", self)

    # --- Cumulative / shift (Polars-style window conveniences) ------------------
    # Each returns a window expression (running aggregate / lag-lead), so use it in
    # `with_columns`/`select`; window expressions do not nest in scalar arithmetic
    # or `filter`. `partition_by` gives a per-group running value; without `order_by`
    # the order is the row order (Polars' default), matching `cum_*` semantics.
    def _running(
        self, agg: str, partition_by: Iterable[IntoExpr], order_by: Iterable[IntoExpr]
    ) -> WindowExpr:
        return AggExpr(agg, self).over(
            partition_by=partition_by, order_by=order_by, frame=(None, 0)
        )

    def cum_sum(
        self, *, partition_by: Iterable[IntoExpr] = (), order_by: Iterable[IntoExpr] = ()
    ) -> WindowExpr:
        """Cumulative (running) sum from the first row to the current one — Polars ``cum_sum``.

        A window expression (one value per row, no row collapse) — use it in
        ``with_columns``/``select``, not in scalar arithmetic or ``filter``. Without
        `order_by` the running order is the row order.

        Args:
            partition_by: Restart the running sum per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A window expression carrying the running sum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.with_columns(cs=bt.col("x").cum_sum()).to_pydict()
                {'x': [1, 2, 3, 4], 'cs': [1, 3, 6, 10]}
        """
        return self._running("sum", partition_by, order_by)

    def cum_min(
        self, *, partition_by: Iterable[IntoExpr] = (), order_by: Iterable[IntoExpr] = ()
    ) -> WindowExpr:
        """Cumulative (running) minimum up to the current row — Polars ``cum_min``.

        A window expression; use it in ``with_columns``/``select``. Pass
        `partition_by` to restart per group and `order_by` to set the running order.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A window expression carrying the running minimum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1, 5]})
                >>> ds.with_columns(cm=bt.col("x").cum_min()).to_pydict()
                {'x': [3, 1, 4, 1, 5], 'cm': [3, 1, 1, 1, 1]}
        """
        return self._running("min", partition_by, order_by)

    def cum_max(
        self, *, partition_by: Iterable[IntoExpr] = (), order_by: Iterable[IntoExpr] = ()
    ) -> WindowExpr:
        """Cumulative (running) maximum up to the current row — Polars ``cum_max``.

        A window expression; use it in ``with_columns``/``select``. Pass
        `partition_by` to restart per group and `order_by` to set the running order.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A window expression carrying the running maximum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1, 5]})
                >>> ds.with_columns(cm=bt.col("x").cum_max()).to_pydict()
                {'x': [3, 1, 4, 1, 5], 'cm': [3, 3, 4, 4, 5]}
        """
        return self._running("max", partition_by, order_by)

    def cum_count(
        self, *, partition_by: Iterable[IntoExpr] = (), order_by: Iterable[IntoExpr] = ()
    ) -> WindowExpr:
        """Cumulative count of non-null values up to the current row — Polars ``cum_count``.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A window expression carrying the running count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1]})
                >>> ds.with_columns(cc=bt.col("x").cum_count()).to_pydict()
                {'x': [3, 1, 4, 1], 'cc': [1, 2, 3, 4]}
        """
        return self._running("count", partition_by, order_by)

    def shift(self, n: int = 1) -> WindowExpr:
        """Shift values by `n` rows in row order — Polars ``shift`` (lag/lead).

        Positive `n` lags (moves down, vacated leading rows null); negative `n` leads
        (moves up). A window expression — use in ``with_columns``/``select``.

        Args:
            n: Number of rows to shift; positive lags, negative leads.

        Returns:
            A window expression with the values shifted by `n` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.with_columns(s=bt.col("x").shift(1)).to_pydict()
                {'x': [1, 2, 3, 4], 's': [None, 1, 2, 3]}
        """
        from batcher.plan.expr_ir.nodes import lag, lead

        return lag(self, n) if n >= 0 else lead(self, -n)

    def forward_fill(self) -> WindowExpr:
        """Carry the last non-null value forward — Polars ``forward_fill``.

        The time-series gap filler: a sensor that reports only on change, a price series
        sampled at irregular times, a slowly-changing dimension. Each row takes the
        nearest non-null value at or before it; rows before the first non-null stay null.

        A window expression, so it must be bound with ``.over(...)`` and **``order_by``
        is required** — a fill carries values along a defined row order, and an
        unordered relation has none. ``partition_by`` keeps each series independent, so
        one device's reading never leaks into another's gap.

        In SQL this is ``last_value(x IGNORE NULLS) OVER (… ROWS UNBOUNDED PRECEDING)``;
        the frame is implied here, so there is none to pass.

        Returns:
            A window expression carrying the forward-filled column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4], "x": [10, None, None, 40]})
                >>> ds.with_columns(f=bt.col("x").forward_fill().over(order_by=["t"])).to_pydict()
                {'t': [1, 2, 3, 4], 'x': [10, None, None, 40], 'f': [10, 10, 10, 40]}
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        return WindowExpr("forward_fill", self, [], [], None)

    def backward_fill(self) -> WindowExpr:
        """Carry the next non-null value backward — Polars ``backward_fill``.

        The mirror of :meth:`forward_fill`: each row takes the nearest non-null value at
        or after it, and rows after the last non-null stay null. ``order_by`` is likewise
        required. Use it to seed a series whose first readings are missing.

        Returns:
            A window expression carrying the backward-filled column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3], "x": [None, None, 30]})
                >>> ds.with_columns(b=bt.col("x").backward_fill().over(order_by=["t"])).to_pydict()
                {'t': [1, 2, 3], 'x': [None, None, 30], 'b': [30, 30, 30]}
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        return WindowExpr("backward_fill", self, [], [], None)

    # --- rolling (fixed-size trailing window) aggregates --------------------
    def _rolling(
        self,
        agg: str,
        window_size: int,
        min_periods: int | None,
        partition_by: Iterable[IntoExpr],
        order_by: Iterable[IntoExpr],
    ) -> Expr:
        """`agg` over the `window_size` rows ending at the current one.

        A ROWS frame of ``(-(window_size - 1), 0)``. Without `min_periods` the leading
        rows of a partition aggregate a *partial* frame, as SQL does. With it, a row
        whose frame holds fewer than `min_periods` non-null values becomes null — the
        guard is a windowed `count` over the same frame, and the null is `nullif` of
        the value against itself (a null of the aggregate's own type). Both compose out
        of existing nodes, so rolling adds no IR."""
        from batcher.plan.expr_ir.constructors import nullif, when

        if window_size < 1:
            raise PlanError(f"rolling_{agg}(): window_size must be >= 1, got {window_size}")
        if min_periods is not None and not 1 <= min_periods <= window_size:
            raise PlanError(
                f"rolling_{agg}(): min_periods must be in [1, {window_size}], got {min_periods}"
            )
        frame = (-(window_size - 1), 0)
        value = AggExpr(agg, self).over(partition_by=partition_by, order_by=order_by, frame=frame)
        if min_periods is None:
            return value
        seen = AggExpr("count", self).over(
            partition_by=partition_by, order_by=order_by, frame=frame
        )
        # `value` is reused in both branches; `hoist_windows` shares the one Window node.
        return when(seen >= Lit(min_periods)).then(value).otherwise(nullif(value, value))

    def rolling_sum(
        self,
        window_size: int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Sum over the `window_size` rows ending at the current one — Polars ``rolling_sum``.

        A window expression; use it in ``with_columns``/``select``. The leading rows of
        each partition aggregate a partial window (SQL semantics); pass `min_periods`
        to make them null instead.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            The rolling sum.

        Raises:
            PlanError: If `window_size` < 1, or `min_periods` is outside
                ``[1, window_size]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_sum(2)).to_pydict()
                {'x': [1, 2, 3, 4], 'r': [1, 3, 5, 7]}
                >>> ds.with_columns(r=bt.col("x").rolling_sum(2, min_periods=2)).to_pydict()
                {'x': [1, 2, 3, 4], 'r': [None, 3, 5, 7]}
        """
        return self._rolling("sum", window_size, min_periods, partition_by, order_by)

    def rolling_mean(
        self,
        window_size: int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Mean over the `window_size` rows ending at the current one — the moving average.

        See :meth:`rolling_sum` for the framing and `min_periods` semantics.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling mean.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_mean(2)).to_pydict()
                {'x': [1, 2, 3, 4], 'r': [1.0, 1.5, 2.5, 3.5]}
        """
        return self._rolling("avg", window_size, min_periods, partition_by, order_by)

    def rolling_min(
        self,
        window_size: int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Minimum over the `window_size` rows ending at the current one.

        See :meth:`rolling_sum` for the framing and `min_periods` semantics.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling minimum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1]})
                >>> ds.with_columns(r=bt.col("x").rolling_min(2)).to_pydict()
                {'x': [3, 1, 4, 1], 'r': [3, 1, 1, 1]}
        """
        return self._rolling("min", window_size, min_periods, partition_by, order_by)

    def rolling_max(
        self,
        window_size: int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Maximum over the `window_size` rows ending at the current one.

        See :meth:`rolling_sum` for the framing and `min_periods` semantics.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling maximum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1]})
                >>> ds.with_columns(r=bt.col("x").rolling_max(2)).to_pydict()
                {'x': [3, 1, 4, 1], 'r': [3, 3, 4, 4]}
        """
        return self._rolling("max", window_size, min_periods, partition_by, order_by)

    def rolling_count(
        self,
        window_size: int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Count of non-null values over the `window_size` rows ending at the current one.

        See :meth:`rolling_sum` for the framing and `min_periods` semantics.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_count(2)).to_pydict()
                {'x': [1, None, 3, 4], 'r': [1, 1, 1, 2]}
        """
        return self._rolling("count", window_size, min_periods, partition_by, order_by)

    def diff(
        self,
        n: int = 1,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """The change from `n` rows back — Polars ``diff``, SQL ``x - lag(x, n) OVER (…)``.

        A window expression composed with subtraction, so the first `n` rows of each
        partition are null. Use it in ``with_columns``/``select``.

        Args:
            n: How many rows back to compare against; negative looks forward.
            partition_by: Restart the comparison per group of these key expressions.
            order_by: Order rows by these expressions before comparing.

        Returns:
            The difference between each value and the one `n` rows away.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 3, 8]})
                >>> ds.with_columns(d=bt.col("x").diff()).to_pydict()
                {'x': [1, 3, 8], 'd': [None, 2, 5]}
        """
        return self - self.shift(n).over(partition_by=partition_by, order_by=order_by)

    def pct_change(
        self,
        n: int = 1,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """The fractional change from `n` rows back — Polars ``pct_change``.

        ``x / lag(x, n) - 1``, evaluated as true division, so integer columns yield a
        float. The first `n` rows of each partition are null.

        Args:
            n: How many rows back to compare against; negative looks forward.
            partition_by: Restart the comparison per group of these key expressions.
            order_by: Order rows by these expressions before comparing.

        Returns:
            The relative change from the value `n` rows away (``0.5`` == +50%).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [10, 15, 30]})
                >>> ds.with_columns(p=bt.col("x").pct_change()).to_pydict()
                {'x': [10, 15, 30], 'p': [None, 0.5, 1.0]}
        """
        return self / self.shift(n).over(partition_by=partition_by, order_by=order_by) - 1

    def fill_nan(self, value: IntoExpr) -> Expr:
        """Replace IEEE NaN with `value`, leaving nulls and ordinary numbers alone.

        The NaN counterpart of :meth:`fill_null`: NaN is a float value, not a null, so
        ``fill_null`` never touches it. A null input stays null.

        Args:
            value: The replacement used wherever this expression is NaN.

        Returns:
            An expression with every NaN replaced by `value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("nan"), 3.0]})
                >>> ds.select(r=bt.col("x").fill_nan(0.0)).to_pydict()
                {'r': [1.0, 0.0, 3.0]}
        """
        from batcher.plan.expr_ir.constructors import when

        return when(self.is_nan()).then(_wrap(value)).otherwise(self)

    def cut(
        self,
        breaks: Iterable[float],
        labels: Iterable[str] | None = None,
        *,
        left_closed: bool = False,
    ) -> Expr:
        """Bin a numeric column into labelled intervals — Polars ``cut``, pandas ``cut``.

        The move from a measurement to a category: ages to cohorts, latencies to SLA
        buckets, scores to grades. `breaks` are the interior boundaries, so `n` breaks
        make `n + 1` bins, and the outermost two are unbounded.

        Bins are right-closed by default — ``(-inf, b0]``, ``(b0, b1]``, …,
        ``(bn, inf]`` — matching Polars and pandas. Pass ``left_closed=True`` for
        ``[-inf, b0)``, ``[b0, b1)``, …, ``[bn, inf)``. A null input yields a null bin
        rather than falling into the last one.

        This lowers to a `CASE` chain over existing IR, so it adds no plan node and runs
        in the Rust expression evaluator like any other projection.

        Args:
            breaks: Interior bin boundaries, strictly increasing.
            labels: One name per bin (``len(breaks) + 1`` of them). Defaults to the
                interval notation, e.g. ``"(1, 5]"``.
            left_closed: Close each interval on the left instead of the right.

        Returns:
            A Utf8 expression carrying each row's bin label.

        Raises:
            PlanError: If `breaks` is empty or not strictly increasing, or if `labels`
                does not have exactly ``len(breaks) + 1`` entries.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"age": [7, 21, 64, None]})
                >>> bins = bt.col("age").cut([12, 19, 65], ["child", "teen", "adult", "senior"])
                >>> ds.select(cohort=bins).to_pydict()
                {'cohort': ['child', 'adult', 'adult', None]}

                >>> ds.select(b=bt.col("age").cut([12, 19])).to_pydict()
                {'b': ['(-inf, 12]', '(19, inf]', '(19, inf]', None]}
        """
        from batcher.plan.expr_ir.constructors import lit, nullif, when

        edges = [float(b) for b in breaks]
        if not edges:
            raise PlanError("cut(): breaks must not be empty")
        if any(lo >= hi for lo, hi in itertools.pairwise(edges)):
            raise PlanError(f"cut(): breaks must be strictly increasing, got {edges}")
        names = list(labels) if labels is not None else _cut_labels(edges, left_closed)
        if len(names) != len(edges) + 1:
            raise PlanError(
                f"cut(): {len(edges)} breaks make {len(edges) + 1} bins, "
                f"but {len(names)} label(s) were given"
            )
        # A null value makes every comparison null, so without this guard it would fall
        # through the CASE chain into the final `otherwise` and be labelled as the top
        # bin. NaN needs the same guard: the engine's total order ranks it above every
        # edge, so it too would land in the top bin, but Polars/pandas leave it null.
        # `nullif(x, x)` is a null of the label column's own type.
        builder = when(self.is_null() | self.is_nan()).then(nullif(lit(names[0]), lit(names[0])))
        for edge, name in zip(edges, names, strict=False):
            below = self < lit(edge) if left_closed else self <= lit(edge)
            builder = builder.when(below).then(lit(name))
        return builder.otherwise(lit(names[-1]))

    def rank(
        self,
        method: str = "min",
        *,
        descending: bool = False,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Rank the rows by this expression's value — SQL ``RANK() OVER (ORDER BY self)``.

        A window expression; use it in ``with_columns``/``select``. Ranks start at 1.

        Args:
            method: How ties are numbered. ``"min"`` gives tied rows the same rank and
                leaves a gap (SQL ``RANK``); ``"dense"`` gives the same rank with no gap
                (``DENSE_RANK``); ``"ordinal"`` breaks ties arbitrarily so every row gets
                a distinct rank (``ROW_NUMBER``).
            descending: Rank from the largest value down instead of the smallest up.
            partition_by: Rank within each group of these key expressions.

        Returns:
            The 1-based rank of each row.

        Raises:
            PlanError: If `method` is not one of ``min``/``dense``/``ordinal``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [10, 30, 10]})
                >>> ds.with_columns(r=bt.col("x").rank()).to_pydict()
                {'x': [10, 30, 10], 'r': [1, 3, 1]}
        """
        from batcher.plan.expr_ir.nodes import dense_rank, rank, row_number

        fns = {"min": rank, "dense": dense_rank, "ordinal": row_number}
        if method not in fns:
            raise PlanError(f"rank(): method must be one of {sorted(fns)}, got {method!r}")
        return fns[method]().over(partition_by=partition_by, order_by=[(self, descending)])

    def is_duplicated(self) -> Expr:
        """True on every row whose value occurs more than once — Polars ``is_duplicated``.

        A window expression (``count(*) OVER (PARTITION BY self) > 1``); use it in
        ``with_columns``/``select``/``filter``. Nulls form their own group, so repeated
        nulls are duplicates.

        Returns:
            A boolean window expression, true on duplicated rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 1]})
                >>> ds.with_columns(d=bt.col("x").is_duplicated()).to_pydict()
                {'x': [1, 2, 1], 'd': [True, False, True]}
        """
        return self._value_count() > Lit(1)

    def is_unique(self) -> Expr:
        """True on every row whose value occurs exactly once — negation of :meth:`is_duplicated`.

        Returns:
            A boolean window expression, true where the value is unique.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 1]})
                >>> ds.with_columns(u=bt.col("x").is_unique()).to_pydict()
                {'x': [1, 2, 1], 'u': [False, True, False]}
        """
        return self._value_count() == Lit(1)

    def _value_count(self) -> WindowExpr:
        """``count(1) OVER (PARTITION BY self)`` — how often each value occurs.

        Counts *rows*, not non-null values: the argument is a literal so a partition of
        nulls still counts its own rows (nulls group together, as in Polars). Counting
        `self` instead would report 0 for every null row."""
        return AggExpr("count", Lit(1)).over(partition_by=[self])


# Imported here, after `Expr` is defined, to break the import cycle: `node_base`
# needs `Expr` as its base class, and the concrete nodes below need `node_base`.
# By the time this line runs, `Expr` is bound, so `node_base`'s top-level
# `from ...core import Expr` resolves against this partially-initialized module.
from batcher.plan.expr_ir.fn_names import MATH_FNS, Math2Fn  # noqa: E402
from batcher.plan.expr_ir.node_base import (  # noqa: E402
    IRNode,
    child,
    children,
    expr_node,
    scalar,
)


class Lit(Expr):
    """A constant literal. The wire kind is inferred from the Python type."""

    __slots__ = ("value",)

    def __init__(self, value: int | float | bool | str) -> None:
        """Wrap a Python scalar (or date/datetime) as a literal expression node."""
        self.value = value

    def to_ir(self) -> dict[str, Any]:
        """Lower this literal to its JSON IR dict (the Rust wire contract)."""
        import datetime as _dt

        v = self.value
        # bool must be checked before int (bool is a subclass of int); likewise
        # datetime before date (datetime subclasses date).
        if isinstance(v, bool):
            tagged = {"bool": v}
        elif isinstance(v, int):
            tagged = {"int": v}
        elif isinstance(v, float):
            # JSON has no NaN/Infinity tokens, and serde_json rejects the
            # non-standard ones Python's ``json.dumps`` would emit — so a
            # ``lit(float("nan"))`` / ``lit(inf)`` used to fail plan parsing
            # entirely. Encode a non-finite float as a name string the Rust
            # ``Literal::Float`` deserializer understands; finite floats stay
            # numeric (unchanged wire, fast path).
            if v != v:
                tagged = {"float": "NaN"}
            elif v == float("inf"):
                tagged = {"float": "inf"}
            elif v == float("-inf"):
                tagged = {"float": "-inf"}
            else:
                tagged = {"float": v}
        elif isinstance(v, str):
            tagged = {"str": v}
        elif isinstance(v, _dt.datetime):
            # Microseconds since the Unix epoch, naive = wall clock (matches how
            # pyarrow stores tz-naive Timestamp(us) columns).
            delta = v - _dt.datetime(1970, 1, 1)
            micros = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
            tagged = {"timestamp": micros}
        elif isinstance(v, _dt.date):
            tagged = {"date": (v - _dt.date(1970, 1, 1)).days}
        else:  # pragma: no cover - guarded by typing
            raise TypeError(f"unsupported literal type: {type(v).__name__}")
        return {"e": ExprTag.LIT, "value": tagged}


@expr_node
class Binary(IRNode):
    """A binary operation over two sub-expressions."""

    tag = ExprTag.BINARY
    op: str = scalar()
    left: Expr = child()
    right: Expr = child()


class InList(Expr):
    """`input IN (values)` — membership in a constant set (the folded form of an
    `(x = v0) OR (x = v1) OR …` chain). `values` are Python scalars of one type
    (int / str / date) matching the input column; lowered to a hash-set lookup."""

    __slots__ = ("input", "values")

    def __init__(self, input: Expr, values: tuple) -> None:
        """Wrap a membership test over a constant `values` set."""
        self.input = input
        self.values = tuple(values)

    def to_ir(self) -> dict[str, Any]:
        """Lower to ``{"e": "in_list", "input": …, "set": [<tagged literal>, …]}``."""
        return {
            "e": ExprTag.IN_LIST,
            "input": self.input.to_ir(),
            "set": [Lit(v).to_ir()["value"] for v in self.values],
        }


@expr_node
class Not(IRNode):
    """Logical negation of a boolean sub-expression."""

    tag = ExprTag.NOT
    input: Expr = child()


@expr_node
class Cast(IRNode):
    """Cast a sub-expression to a named Arrow type.

    `try_cast` selects DuckDB ``TRY_CAST`` semantics — a value that cannot be
    converted yields NULL instead of erroring the query; the default strict
    ``CAST`` errors on an invalid value.
    """

    tag = ExprTag.CAST
    input: Expr = child()
    dtype: str = scalar()
    try_cast: bool = scalar(default=False)


@expr_node
class IsNull(IRNode):
    """True where the argument is null."""

    tag = ExprTag.IS_NULL
    input: Expr = child()


@expr_node
class IsNotNull(IRNode):
    """True where the argument is non-null."""

    tag = ExprTag.IS_NOT_NULL
    input: Expr = child()


@expr_node
class IsNan(IRNode):
    """True where a float value is IEEE NaN (null → null)."""

    tag = ExprTag.IS_NAN
    input: Expr = child()


@expr_node
class IsInf(IRNode):
    """True where a float value is ``+inf`` or ``-inf`` (null → null)."""

    tag = ExprTag.IS_INF
    input: Expr = child()


class Aliased(Expr):
    """An expression tagged with an output name (from `Expr.alias`).

    Transparent in the IR — `to_ir` delegates to the wrapped expression, so the
    name is carried only at the API/projection boundary. Reachable via
    `Expr.alias(name)`; not constructed directly.
    """

    __slots__ = ("inner", "name")

    def __init__(self, inner: Expr, name: str) -> None:
        """Wrap an expression with an output name (built by :meth:`Expr.alias`)."""
        self.inner = inner
        self.name = name

    def to_ir(self) -> dict[str, Any]:
        """Lower to the wrapped expression's JSON IR (the alias is transparent in the IR)."""
        return self.inner.to_ir()


def normalize_key_list(keys: IntoExpr | Iterable[IntoExpr]) -> list[IntoExpr]:
    """Normalize a ``partition_by``/``order_by`` argument to a list of key expressions.

    A single ``str`` column name or a lone ``Expr`` is wrapped in a one-element list; an
    existing iterable of keys is materialized with ``list``. Without this, the natural
    scalar spellings silently corrupt: ``over(partition_by="grp")`` would ``list("grp")``
    into ``['g', 'r', 'p']`` (partition by three phantom columns), and
    ``over(partition_by=col("g"))`` would iterate an `Expr` — which has an unbounded
    `__getitem__` — until memory is exhausted.
    """
    if isinstance(keys, (str, Expr)):
        return [keys]
    return list(keys)


class AggExpr:
    """An aggregate over an optional input expression.

    Built via `col(...).sum()` etc. or the top-level `count()`; bound to an output
    name when passed to `group_by(...).agg(name=agg)`. Serializes to the engine's
    `AggregateItem` shape.

    Aggregates come in three shapes, distinguished by the keyword-only arguments:
    *unary* (`sum`, `mean`, …) take just `input`; *binary* (`corr`, `covar_*`,
    `arg_min`, `arg_max`) take a second expression via `input2`; *parametric*
    (`quantile`, `approx_quantile`) take a scalar via `param`. The two are
    keyword-only so a call site can never silently swap the second input for the
    parameter.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
            >>> ds.group_by("g").agg(total=bt.col("x").sum()).sort("g").to_pydict()
            {'g': ['a', 'b'], 'total': [3, 3]}
    """

    __slots__ = ("func", "input", "input2", "param")

    def __init__(
        self,
        func: str,
        input: Expr | None,
        *,
        input2: Expr | None = None,
        param: float | None = None,
    ) -> None:
        """Construct an aggregate over an optional input, plus an optional `input2` or `param`."""
        self.func = func
        self.input = input
        # The second input expression — the ordering key for arg_min/arg_max or the
        # paired column for corr/covar; None for unary and parametric aggregates.
        self.input2 = input2
        # The scalar parameter for parametric aggregates (the q of quantile); None otherwise.
        self.param = param

    def __repr__(self) -> str:
        """A source-like rendering, e.g. ``col('x').sum()`` or ``count()``."""
        args = []
        if self.input2 is not None:
            args.append(repr(self.input2))
        if self.param is not None:
            args.append(repr(self.param))
        call = f"{self.func}({', '.join(args)})"
        return call if self.input is None else f"{self.input!r}.{call}"

    def to_ir(self, alias: str) -> dict[str, Any]:
        """Lower this aggregate to its JSON ``AggregateItem`` dict, bound to `alias`.

        Args:
            alias: The output column name to bind this aggregate to.

        Returns:
            The aggregate's JSON ``AggregateItem`` dict.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("x").sum().to_ir("total")
                {'func': 'sum', 'alias': 'total', 'input': {'e': 'col', 'name': 'x'}}
        """
        item: dict[str, Any] = {"func": self.func, "alias": alias}
        if self.input is not None:
            item["input"] = self.input.to_ir()
        if self.input2 is not None:
            item["input2"] = self.input2.to_ir()
        if self.param is not None:
            item["param"] = self.param
        return item

    def over(
        self,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        frame: tuple[int | None, int | None] | None = None,
    ):
        """Turn this aggregate into a window expression — SQL ``<agg> OVER (…)``.

        ``col("x").sum().over(partition_by=["g"])`` computes the per-partition sum
        broadcast to every row (no grouping/row collapse). With `order_by` it becomes
        a running aggregate; `frame` sets an explicit ``ROWS`` window. Used inside
        `with_columns`, which lowers it to the relational `Window` operator. Only the
        aggregate functions (`sum`/`mean`/`min`/`max`/`count`) support `over`.

        Args:
            partition_by: Key expressions whose groups the aggregate is computed within.
            order_by: Expressions to order rows by, making it a running aggregate.
            frame: An explicit ``ROWS`` frame as ``(preceding, following)`` offsets; ``None`` for
                the default.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 10]})
                >>> w = bt.col("v").sum().over(partition_by=["g"])
                >>> ds.with_columns(total=w).sort("v").to_pydict()
                {'g': ['a', 'a', 'b'], 'v': [1, 2, 10], 'total': [3, 3, 10]}
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        # `mean` is the DataFrame spelling; the window engine names the aggregate `avg`.
        func = "avg" if self.func == "mean" else self.func
        return WindowExpr(
            func,
            self.input,
            normalize_key_list(partition_by),
            normalize_key_list(order_by),
            frame,
        )


@expr_node
class MathExpr(IRNode):
    """A unary math function over a numeric sub-expression."""

    tag = ExprTag.MATH
    vocab = MATH_FNS
    fn: str = scalar()
    input: Expr = child()


@expr_node
class Math2Expr(IRNode):
    """A two-argument math function (pow/atan2/round-to-digits) → Float64."""

    tag = ExprTag.MATH2
    vocab = frozenset(Math2Fn)
    fn: str = scalar()
    left: Expr = child()
    right: Expr = child()


@expr_node
class Coalesce(IRNode):
    """First non-null among the sub-expressions (SQL COALESCE)."""

    tag = ExprTag.COALESCE
    inputs: list[Expr] = children()
