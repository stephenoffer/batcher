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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import ExprTag
from batcher.plan.types import CAST_DTYPES

if TYPE_CHECKING:
    from batcher.plan.expr_ir.image import _ImageNamespace
    from batcher.plan.expr_ir.namespaces import (
        _DtNamespace,
        _JsonNamespace,
        _ListNamespace,
        _StrNamespace,
        _StructNamespace,
    )
    from batcher.plan.expr_ir.nodes import WindowExpr

# A value that can be promoted to an expression: another Expr or a Python scalar.
IntoExpr = Union["Expr", int, float, bool, str]


def _wrap(value: IntoExpr) -> Expr:
    return value if isinstance(value, Expr) else Lit(value)


class Expr:
    """Base class for scalar expressions — the one expression type in Batcher.

    An ``Expr`` is an immutable IR node, built lazily with operator overloading and
    fluent methods (``col("x") * 2``, ``col("x").sqrt()``, ``col("g").sum()``) and
    serialized via :meth:`to_ir` to the JSON the Rust ``bc-expr`` engine evaluates —
    no Python touches a row. Methods come in families: arithmetic/comparison/boolean
    operators, math functions (``sqrt``, ``ln``, ``sin``, …), null/NaN predicates
    (``is_null``, ``is_nan``, ``fill_null``), aggregates for ``group_by().agg(...)``
    / ``.over(...)`` (``sum``, ``mean``, ``count``, …), cumulative window helpers
    (``cum_sum``, ``shift``), and the typed accessor namespaces (``.str``, ``.dt``,
    ``.list``, ``.struct``, ``.json``, ``.image``, ``.audio``, ``.video``, ``.map``)
    that hold the per-type breadth.

    Subclasses are the concrete IR nodes (``Lit``, ``Binary``, ``MathExpr``, …); user
    code constructs expressions through ``col``/``lit`` and these methods, not the
    node classes directly.
    """

    # --- serialization -----------------------------------------------------
    def to_ir(self) -> dict[str, Any]:  # pragma: no cover - overridden
        """Serialize this expression to its JSON IR dict — the wire contract with the engine.

        Each node emits ``{"e": <tag>, ...}`` matching the ``bc_expr::Expr`` serde
        tags the Rust interpreter and JIT deserialize. Overridden by every subclass;
        the base raises ``NotImplementedError``. Internal — not part of the user API.
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
        """Element-wise true division (``a / b``, → Float64); ``//`` is :meth:`__floordiv__`."""
        return Binary("div", self, _wrap(other))

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
        return Binary("div", _wrap(other), self)

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

    def __getitem__(self, key: int | slice | str) -> Expr:
        """Index into a list or struct column with ``[]`` — the idiomatic spelling of
        the ``.list``/``.struct`` accessors it delegates to.

        - ``col("a")[2]`` → list element at index 2 (negative counts from the end),
          equivalent to ``col("a").list.get(2)``.
        - ``col("a")[1:3]`` → list sub-range ``[1, 3)``, equivalent to
          ``col("a").list.slice(1, 2)`` (a ``step`` other than 1 raises).
        - ``col("s")["field"]`` → struct field, equivalent to
          ``col("s").struct.field("field")``.
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

    # --- bitwise integer operators (distinct from the boolean `&`/`|`) ------
    def bitwise_and(self, other: IntoExpr) -> Expr:
        """Bitwise AND ``self & other`` of two integer expressions.

        Operates per row on the integer bit patterns (operands cast to Int64), unlike
        the ``&`` operator which is boolean AND on predicates. The method spelling is
        unambiguous; nulls propagate.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.select(r=bt.col("x").cast("float64")).to_pydict()
                {'r': [1.0, 2.0]}
        """
        return self._cast(dtype, try_cast=False)

    def try_cast(self, dtype: str) -> Cast:
        """Cast to an Arrow type by name, yielding NULL for values that cannot be
        converted (DuckDB ``TRY_CAST``) instead of erroring the query.

        The common safe-ingest spelling: ``col("x").try_cast("int64")`` turns a
        dirty string column into integers, with unparseable values becoming NULL
        (ready to `drop_nulls` or route to a quarantine sink).

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.select(r=bt.col("x").is_in([1, 3])).to_pydict()
                {'r': [True, False, True]}
        """
        vals = list(values)
        if not vals:
            return Lit(False)
        expr: Expr = self == vals[0]
        for v in vals[1:]:
            expr = expr | (self == v)
        return expr

    def between(self, low: IntoExpr, high: IntoExpr) -> Expr:
        """``self BETWEEN low AND high`` (inclusive on both bounds), matching SQL/DuckDB.

        Desugars to ``(self >= low) & (self <= high)``, so it follows SQL three-valued
        logic — a null operand makes the result null. The idiomatic spelling for a
        range filter (chained comparisons like ``low <= col("x") <= high`` are
        rejected; see :meth:`__bool__`).

        Args:
            low: Inclusive lower bound.
            high: Inclusive upper bound.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 10]})
                >>> ds.select(r=bt.col("x").between(2, 8)).to_pydict()
                {'r': [False, True, False]}
        """
        return (self >= low) & (self <= high)

    def eq_missing(self, other: IntoExpr) -> Expr:
        """Null-safe equality (SQL ``IS NOT DISTINCT FROM``): two nulls compare
        equal, and a null vs a non-null compares **false** (never null).

        The reliable way to compare possibly-null keys — used for change detection
        in slowly-changing dimensions. Desugars to existing ops (no new IR).

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
        """Remap values via a ``{old: new}`` dictionary (value standardization /
        lookup recode). Values absent from `mapping` keep their original value, or
        take `default` when one is given. Desugars to a ``CASE`` chain (no new IR).

        ``col("c").replace({"US": "USA", "UK": "GBR"})`` standardizes country codes.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0]})
                >>> ds.select(r=bt.col("x").cot()).to_pydict()
                {'r': [0.6420926159343308]}
        """
        return MathExpr("cot", self)

    def factorial(self) -> MathExpr:
        """``n!`` — factorial of a non-negative integer (DuckDB ``factorial``; → Float64).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5]})
                >>> ds.select(f=bt.col("x").factorial()).to_pydict()
                {'f': [120.0]}
        """
        return MathExpr("factorial", self)

    def bit_count(self) -> MathExpr:
        """Population count — the number of set bits in the integer value
        (DuckDB ``bit_count``).

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
    def map(self):
        """Map accessor — grouped key/value access on a map column.

        Returns a namespace with ``.map.keys()``, ``.map.values()``, and
        ``.map.get(key)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> Expr = type(bt.col("x")).__mro__[1]
                >>> isinstance(bt.col("m").map.get("k"), Expr)
                True
        """
        from batcher.plan.expr_ir.namespaces import _MapNamespace

        return _MapNamespace(self)

    @property
    def json(self) -> _JsonNamespace:
        """JSON accessor — grouped JSONPath extraction on a JSON-string column.

        Returns a namespace with typed extractors such as
        ``.json.extract_string("$.a")``, evaluated in the engine (no Python parsing).

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
    def audio(self):
        """Audio accessor — grouped lazy audio-decode ops on a binary column.

        Returns a namespace with ops such as ``.audio.decode()`` and
        ``.audio.to_waveform()``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> Expr = type(bt.col("x")).__mro__[1]
                >>> isinstance(bt.col("a").audio.decode(), Expr)
                True
        """
        from batcher.plan.expr_ir.audio import _AudioNamespace

        return _AudioNamespace(self)

    @property
    def video(self):
        """Video accessor — grouped lazy video-decode ops on a binary column.

        Returns a namespace with ops such as ``.video.decode()`` (requires the engine
        built with the ``video`` feature).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> Expr = type(bt.col("x")).__mro__[1]
                >>> isinstance(bt.col("v").video.decode(), Expr)
                True
        """
        from batcher.plan.expr_ir.video import _VideoNamespace

        return _VideoNamespace(self)

    def fill_null(self, value: IntoExpr) -> Coalesce:
        """Replace nulls with `value`, leaving non-null values unchanged (SQL ``COALESCE``).

        `value` may be a scalar or another expression (e.g. a column to fall back to).
        Only NULL is replaced — float NaN is not a null, so use :meth:`is_nan` to
        handle it.

        Args:
            value: The replacement used wherever this expression is null.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("inf")]})
                >>> ds.select(r=bt.col("x").is_infinite()).to_pydict()
                {'r': [False, True]}
        """
        return IsInf(self)

    def is_finite(self) -> Expr:
        """True where the value is finite — not NaN and not ``±inf`` (Polars/pandas
        ``is_finite``). Nulls propagate (null → null).

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
        through to the original value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 10]})
                >>> ds.select(r=bt.col("x").clip(2, 8)).to_pydict()
                {'r': [2, 5, 8]}
        """
        from batcher.plan.expr_ir.constructors import when

        result: Expr = self
        if lower is not None:
            result = when(result < _wrap(lower)).then(lower).otherwise(result)
        if upper is not None:
            result = when(result > _wrap(upper)).then(upper).otherwise(result)
        return result

    # --- aggregate constructors (used inside group_by().agg(...)) -----------
    def sum(self) -> AggExpr:
        """Sum of non-null values per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        An aggregate: it collapses a group to one row (or, via :meth:`AggExpr.over`,
        broadcasts the group result to each row). Mergeable, so identical single-node
        and distributed.

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 4, 6]})
                >>> ds.group_by("g").agg(r=bt.col("x").std()).to_pydict()
                {'g': ['a'], 'r': [2.0]}
        """
        return AggExpr("stddev", self)

    def skewness(self) -> AggExpr:
        """Sample skewness of this expression's non-null values per group
        (adjusted Fisher-Pearson, matching DuckDB; -> Float64). Null when the group
        has fewer than 3 values. Mergeable (sum-of-powers moment state).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").skewness()).to_pydict()
                {'g': ['a'], 'r': [1.763632614803888]}
        """
        return AggExpr("skewness", self)

    def kurtosis(self) -> AggExpr:
        """Sample excess kurtosis per group (0 for a normal distribution, matching
        DuckDB; → Float64). Null when the group has fewer than 4 values. Mergeable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").kurtosis()).to_pydict()
                {'g': ['a'], 'r': [3.152000000000008]}
        """
        return AggExpr("kurtosis", self)

    def median(self) -> AggExpr:
        """Exact median per group — the 0.5 quantile, averaging the two middle values
        for an even count (→ Float64). Equals ``quantile(0.5)``. An aggregate for
        ``group_by().agg(...)``; see :meth:`approx_median` for a bounded-memory sketch.

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
        """Approximate median (the 0.5 quantile) via a KLL sketch — see
        `approx_quantile`.

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
        """Most frequent value per group. Ties are broken by the smallest value
        (deterministic and partition-independent). Works on any column type.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").mode()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("mode", self)

    def first(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the first row in `order_by` order (SQL
        ``first(x ORDER BY order_by)``). Equivalent to ``arg_min(order_by)``.

        An explicit `order_by` is **required**: an arrival-order first/last is not
        partition-independent, so it could not stay identical single-node and
        distributed. With an order key the result is deterministic and mergeable
        (ties on the key break to the smallest value).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").first(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        return AggExpr("arg_min", self, input2=_wrap(order_by))

    def last(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the last row in `order_by` order (SQL
        ``last(x ORDER BY order_by)``). Equivalent to ``arg_max(order_by)``. As with
        :meth:`first`, an explicit `order_by` is required so the result stays
        deterministic and mergeable across partitions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").last(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("arg_max", self, input2=_wrap(order_by))

    def arg_min(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is minimal in the group
        (SQL ``arg_min``/``min_by``). Key ties break to the smallest value, so the
        result is deterministic and partition-independent.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").arg_min(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        return AggExpr("arg_min", self, input2=_wrap(by))

    def arg_max(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is maximal in the group
        (SQL ``arg_max``/``max_by``).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").arg_max(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        return AggExpr("arg_max", self, input2=_wrap(by))

    def bool_and(self) -> AggExpr:
        """Logical AND of this boolean expression's non-null values per group
        (null when the group has no non-null value).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, True]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_and()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [False, True]}
        """
        return AggExpr("bool_and", self)

    def bool_or(self) -> AggExpr:
        """Logical OR of this boolean expression's non-null values per group
        (null when the group has no non-null value).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, False]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_or()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [True, False]}
        """
        return AggExpr("bool_or", self)

    def product(self) -> AggExpr:
        """Product of this expression's non-null values per group (DuckDB
        ``product``; → Float64). Mergeable, so identical single-node and distributed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").product()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [6.0, 5.0]}
        """
        return AggExpr("product", self)

    def bit_and(self) -> AggExpr:
        """Bitwise AND of this expression's non-null Int64 values per group
        (Spark/DuckDB ``bit_and``). Mergeable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_and()).to_pydict()
                {'g': ['a'], 'r': [2]}
        """
        return AggExpr("bit_and", self)

    def bit_or(self) -> AggExpr:
        """Bitwise OR of this expression's non-null Int64 values per group
        (Spark/DuckDB ``bit_or``). Mergeable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_or()).to_pydict()
                {'g': ['a'], 'r': [7]}
        """
        return AggExpr("bit_or", self)

    def bit_xor(self) -> AggExpr:
        """Bitwise XOR of this expression's non-null Int64 values per group
        (Spark/DuckDB ``bit_xor``). Mergeable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [6, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").bit_xor()).to_pydict()
                {'g': ['a'], 'r': [5]}
        """
        return AggExpr("bit_xor", self)

    def histogram(self) -> AggExpr:
        """Collect this expression's non-null values per group into a
        ``Map<value, count>`` (DuckDB ``histogram``). Keys are the distinct values
        sorted ascending; values are their counts. Mergeable, so identical
        single-node and distributed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 2]})
                >>> ds.group_by("g").agg(r=bt.col("x").histogram()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [[(1, 2)], [(2, 1)]]}
        """
        return AggExpr("histogram", self)

    def array_agg(self) -> AggExpr:
        """Collect this expression's non-null values in each group into a ``List``
        (SQL ``array_agg``; Spark ``collect_list``). Without an explicit order the
        element order is arrival-dependent. Mergeable — the per-group value list is
        the partial state, so the result is the same single-node and distributed.

        Chain a list reduction on the result column to summarize it, e.g.
        ``ds.group_by("g").agg(tags=col("t").array_agg())`` then
        ``col("tags").list.join(",")``.

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
        """Cumulative count of non-null values up to the current row — Polars
        ``cum_count``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 4, 1]})
                >>> ds.with_columns(cc=bt.col("x").cum_count()).to_pydict()
                {'x': [3, 1, 4, 1], 'cc': [1, 2, 3, 4]}
        """
        return self._running("count", partition_by, order_by)

    def shift(self, n: int = 1) -> WindowExpr:
        """Shift values by `n` rows in row order (Polars ``shift``): positive `n` lags
        (moves down, vacated leading rows null), negative `n` leads (moves up). A
        window expression — use in ``with_columns``/``select``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.with_columns(s=bt.col("x").shift(1)).to_pydict()
                {'x': [1, 2, 3, 4], 's': [None, 1, 2, 3]}
        """
        from batcher.plan.expr_ir.nodes import lag, lead

        return lag(self, n) if n >= 0 else lead(self, -n)


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

    def to_ir(self, alias: str) -> dict[str, Any]:
        """Lower this aggregate to its JSON ``AggregateItem`` dict, bound to `alias`."""
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
        return WindowExpr(func, self.input, list(partition_by), list(order_by), frame)


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
