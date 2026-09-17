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

import datetime as _dt
import decimal as _decimal
import itertools
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NoReturn, Union

from batcher._internal.errors import PlanError, require_float, require_int
from batcher._internal.mathx import is_nan
from batcher.plan.expr_ir.compat import expr_attribute_error as _expr_attribute_error
from batcher.plan.ir_tags import MICROS_PER_DAY, ExprTag
from batcher.plan.types import (
    CAST_DTYPES,
    canonical_dtype_name,
    normalize_dtype_spec,
    resolve_dtype,
)

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
    from batcher.plan.expr_ir.namespaces.sequence import _SeqNamespace
    from batcher.plan.expr_ir.nodes import WindowExpr
    from batcher.plan.expr_ir.video import _VideoNamespace

# A value that can be promoted to an expression: another Expr or a Python scalar.
IntoExpr = Union["Expr", int, float, bool, str]


def _wrap(value: IntoExpr) -> Expr:
    # `AggExpr` is not an `Expr` but can be a leaf of one (``col("x").sum() / 2``);
    # pass it through rather than lifting it to a `Lit`. `group_by().agg()` splits such
    # leaves back out; any that reach `to_ir()` elsewhere raise a clear error there.
    if isinstance(value, (Expr, AggExpr)):
        return value  # type: ignore[return-value]
    # A CASE builder is the one non-`Expr` users hand us as an expression on purpose: a
    # ``when(...).then(...)`` without ``.otherwise`` is SQL's ``CASE ... END``, NULL where
    # nothing matched. It finishes into a `Case` here. Matched by name to avoid importing
    # `nodes` (which imports this module).
    if type(value).__name__ == "CaseBuilder":
        return value._finish()  # type: ignore[attr-defined]
    return Lit(value)


# `constructors` imports this module, so `col` is resolved on first use rather than at
# module level — and remembered, instead of re-imported per coerced ordering argument.
_COL = None


def _col_or_expr(value: IntoExpr) -> Expr:
    """An ordering/source argument: a bare string names a *column*, not a string literal.

    ``_wrap`` would turn ``arg_max(v, "k")`` into an ordering by the constant ``'k'``;
    an ``Expr`` passes through unchanged. Mirrors SQL ``arg_max(v, k)`` / DuckDB.
    """
    if isinstance(value, str):
        global _COL
        if _COL is None:
            from batcher.plan.expr_ir.constructors import col

            _COL = col
        return _COL(value)
    return _wrap(value)


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


# Accessor-namespace classes, resolved on first use and remembered. The namespace modules
# import `Expr` from here, so this module cannot import them at module level — but the
# deferred `from ... import ...` each accessor carried then ran on *every* `.str` / `.dt` /
# `.list` / ... access, and a repeat `from X import Y` still costs ~400 ns against ~70 ns
# for a cached lookup. Resolving once keeps the import cycle broken and takes the import
# machinery off the accessor path, which is the widest part of the expression API.
_ACCESSORS: dict[str, type] = {}

# `render` imports this module, so `render_expr` is another name that cannot be imported at
# module level and was therefore re-imported on every `repr()` — which the aggregate-leaf
# registry uses as its dedup key, so it is not only a debugging path.
_RENDER = None


def _render():
    """The `render_expr` function, imported at most once."""
    global _RENDER
    if _RENDER is None:
        from batcher.plan.expr_ir.render import render_expr

        _RENDER = render_expr
    return _RENDER


def _accessor(module: str, name: str) -> type:
    """The accessor-namespace class `name` from `module`, imported at most once."""
    cls = _ACCESSORS.get(name)
    if cls is None:
        import importlib

        cls = getattr(importlib.import_module(module), name)
        _ACCESSORS[name] = cls
    return cls


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
    ``.map``, ``.seq``) that hold the per-type breadth.

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

    def __getattr__(self, name: str) -> Any:
        """Raise an `AttributeError` that names the Batcher spelling for an absent idiom.

        Only reached when normal lookup fails, so it never shadows a real method or a
        typed accessor. A pandas/Polars migrant reaches for an expression method Batcher
        spells differently (``.map_elements``, ``.clip_lower``, ``.argmax``) or does not
        have at expression level (``.filter``, ``.value_counts``); the traceback carries
        the mapping — see `batcher.plan.expr_ir.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        # Dunder and private probes (copy/pickle/inspect, subclass instance state) must
        # fail plainly: a decorated failure would turn a routine hasattr into a hard error.
        if name.startswith("_"):
            raise AttributeError(name)
        raise _expr_attribute_error(self, name)

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

    # `Expr` stays unhashable (see `__hash__` below, which raises with the reason).

    def __repr__(self) -> str:
        """A source-like rendering of the expression, e.g. ``(col('x') + lit(1))``."""
        return _render()(self)

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
        """Floor division ``a // b`` — the quotient rounded toward negative infinity.

        These are Polars/Python semantics, deliberately *not* SQL integer division,
        which truncates toward zero: ``-7 // 3`` is ``-3`` here, where ``-7 / 3``
        gives ``-2``.

        The operation is **type-preserving for integers**: Int64 ``//`` Int64 stays
        Int64 and is computed exactly, including above 2^53 where routing through
        Float64 would silently lose precision. A zero divisor yields NULL on **both**
        arms, matching DuckDB's ``//`` (``1.0 // 0.0`` is NULL there, not ``inf``);
        otherwise a float pair gives ``floor(a / b)``."""
        return Binary("floor_div", self, _wrap(other))

    def __rfloordiv__(self, other: IntoExpr) -> Expr:
        """Reflected floor division so ``scalar // expr`` works; see :meth:`__floordiv__`."""
        return Binary("floor_div", _wrap(other), self)

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
        """Exclusive or ``a ^ b``: boolean over two booleans, else bitwise over Int64.

        Two boolean operands give a boolean (null where either is null), as Polars and
        Python do; integer operands are cast to Int64 and xor bit by bit, the operator
        spelling of :meth:`bitwise_xor`."""
        return Binary("bit_xor", self, _wrap(other))

    def __lshift__(self, other: IntoExpr) -> Expr:
        """Left shift ``a << b``; the operator spelling of :meth:`bitwise_left_shift`."""
        return Binary("shift_left", self, _wrap(other))

    def __rshift__(self, other: IntoExpr) -> Expr:
        """Right shift ``a >> b``; the operator spelling of :meth:`bitwise_right_shift`."""
        return Binary("shift_right", self, _wrap(other))

    def __rxor__(self, other: IntoExpr) -> Expr:
        """Reflected XOR so ``scalar ^ expr`` works (boolean or bitwise, as :meth:`__xor__`)."""
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

    def __iter__(self) -> NoReturn:
        """Refuse iteration: an expression is a scalar column, not a sequence.

        `__getitem__` accepts an int index (``col("a")[2]`` → list element), which makes an
        expression *look* iterable to ``list(expr)`` / ``for x in expr`` — but the index has
        no upper bound (every ``expr[i]`` yields a fresh node), so the default iteration
        protocol would loop forever and exhaust memory. Raising here turns any accidental
        ``list(expr)`` (e.g. ``over(partition_by=col("g"))``) into an immediate, clear error.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> list(bt.col("a"))
                Traceback (most recent call last):
                    ...
                TypeError: a batcher expression is not iterable; wrap it in a list ...

        Raises:
            TypeError: Always — naming the list-wrapping fix.
        """
        raise TypeError(
            "a batcher expression is not iterable; wrap it in a list "
            "(e.g. over(partition_by=[col('g')]), not over(partition_by=col('g'))). "
            "For a row-wise minimum/maximum across columns use least(a, b) / "
            "greatest(a, b); for a column aggregate use .min() / .max()"
        )

    def __len__(self) -> NoReturn:
        """Refuse `len`: an expression describes a column, it does not hold one yet.

        Nothing is materialized until a terminal op, so an expression has no row
        count to report. The row count of the *result* is ``ds.count()``; the length
        of a string or list value is ``.str.len()`` / ``.list.len()``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> len(bt.col("x"))
                Traceback (most recent call last):
                    ...
                TypeError: a batcher expression has no len()...

        Raises:
            TypeError: Always — naming the three things `len` is usually reaching for.
        """
        raise TypeError(
            "a batcher expression has no len(): it describes a column, it does not hold "
            "one. Use .str.len() for string length, .list.len() for list length, or "
            "ds.count() for the number of rows in the result"
        )

    def __contains__(self, item: object) -> NoReturn:
        """Refuse ``x in expr``: the membership operators are `is_in` and `str.contains`.

        Python coerces the result of ``in`` to a bool, so it could never return an
        expression. Without this, ``1 in col("x")`` falls through to `__iter__` and
        raises a confusing "not iterable" message for what is really a membership test.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> 1 in bt.col("x")
                Traceback (most recent call last):
                    ...
                TypeError: `in` cannot be used on a batcher expression...

        Args:
            item: The value the caller tried to test for membership.

        Raises:
            TypeError: Always — naming `is_in` / `str.contains` / `list.contains`.
        """
        raise TypeError(
            "`in` cannot be used on a batcher expression (Python forces the result to a "
            "bool). Use col('x').is_in([...]) to test the value against a set, "
            "col('s').str.contains(...) for substring search, or "
            "col('l').list.contains(...) for list membership"
        )

    def __hash__(self) -> NoReturn:
        """Refuse hashing: ``==`` builds an expression, so equality-based lookup is a trap.

        A hash-based container compares candidates with ``==``, which here returns a
        *predicate* rather than a bool — so a set or dict keyed on expressions would
        silently misbehave. Batcher raises instead, matching pandas and Polars, whose
        expression/series types are likewise unhashable. Key on the column name, or on
        ``to_ir()`` for a structural key.

        Raises:
            TypeError: Always — naming the two workable keys.
        """
        raise TypeError(
            "a batcher expression is not hashable, because `==` builds a predicate "
            "instead of comparing. Key on the column name, or on repr(expr) / "
            "str(expr.to_ir()) for a structural key"
        )

    def __divmod__(self, other: IntoExpr) -> tuple[Expr, Expr]:
        """``divmod(a, b)`` — the ``(a // b, a % b)`` pair, as Python defines it.

        Args:
            other: The divisor value or expression.

        Returns:
            A ``(quotient, remainder)`` tuple of expressions.
        """
        return self // other, self % other

    def __rdivmod__(self, other: IntoExpr) -> tuple[Expr, Expr]:
        """Reflected `divmod` so ``divmod(scalar, expr)`` works.

        Args:
            other: The dividend value or expression.

        Returns:
            A ``(quotient, remainder)`` tuple of expressions.
        """
        return _wrap(other) // self, _wrap(other) % self

    def __matmul__(self, other: IntoExpr) -> Expr:
        """``a @ b`` — the dot product of two list (embedding) columns.

        The numpy spelling of :meth:`_ListNamespace.dot`, which is what an embedding
        similarity reads as: ``col("emb") @ col("query")``.

        Args:
            other: The other list column.

        Returns:
            A Float64 expression of the per-row dot product.
        """
        return self.list.dot(other)

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

        Over two booleans it is the boolean exclusive-or, as ``^`` is.

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
    def cast(self, dtype: str | type) -> Cast:
        """Cast to an Arrow type (int64/float64/int32/bool/string/...).

        The dtype is validated at plan-build time; anything that is not a dtype raises
        rather than failing opaquely in the engine mid-query. A value that cannot be
        converted errors the query (DuckDB ``CAST``); use `try_cast` to get NULL instead.

        Args:
            dtype: Target Arrow type name (e.g. ``"int64"``), a Python type (``int``,
                ``float``, ``str``, ``bool``), or a pyarrow `DataType`.

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

    def try_cast(self, dtype: str | type) -> Cast:
        """Cast to an Arrow type by name; unconvertible values become NULL (DuckDB ``TRY_CAST``).

        The common safe-ingest spelling: ``col("x").try_cast("int64")`` turns a
        dirty string column into integers, with unparseable values becoming NULL
        (ready to `drop_nulls` or route to a quarantine sink).

        Args:
            dtype: Target Arrow type name (e.g. ``"int64"``), a Python type (``int``,
                ``float``, ``str``, ``bool``), or a pyarrow `DataType`.

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

    def _cast(self, dtype: str | type, *, try_cast: bool) -> Cast:
        # Type names are matched case-insensitively (pandas spells these `"Int64"`, SQL
        # `"BIGINT"`, and a case mismatch is a typo the user cannot see), and the IR always
        # carries the canonical form, so the wire contract is unaffected.
        name = normalize_dtype_spec(dtype, caller="try_cast" if try_cast else "cast")
        canonical = canonical_dtype_name(name)
        # `resolve_dtype`, not `canonical in CAST_DTYPES`: the fixed names are only half
        # the vocabulary, and membership-testing the set rejects every parametrized dtype
        # (`decimal(12,4)`, `timestamp(ns)`) that the engine itself accepts.
        if resolve_dtype(canonical) is None:
            import difflib

            hint = difflib.get_close_matches(canonical, sorted(CAST_DTYPES), n=2, cutoff=0.5)
            suffix = f"; did you mean {' or '.join(map(repr, hint))}?" if hint else ""
            raise PlanError(
                f"unknown cast dtype {dtype!r}; valid: {sorted(CAST_DTYPES)}, or a "
                f"parametrized name such as 'decimal(12,4)', 'timestamp(ns)', "
                f"'timestamp(us, UTC)', 'time64(ns)', 'duration(s)'{suffix}"
            )
        return Cast(self, canonical, try_cast=try_cast)

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

    def is_in(self, values: Iterable[IntoExpr], *, nulls_equal: bool = False) -> Expr:
        """``self IN (values)`` — true if equal to any value.

        Desugars to an OR of equality checks, so by default it follows SQL three-valued
        logic: ``NULL IN (...)`` is NULL, a ``None`` among `values` turns every non-match
        into NULL, and an empty collection is always false.

        `nulls_equal=True` is the null-safe reading of Polars ``is_in(nulls_equal=True)``
        and Ray Data ``is_in``: a null matches a ``None`` in `values`, and every other
        answer is ``True`` or ``False``, never null. Its negation ``~x.is_in(v,
        nulls_equal=True)`` is Ray Data's ``not_in``.

        Args:
            values: The scalars or expressions to test membership against.
            nulls_equal: Treat null as a value that equals ``None``, so the result is
                never null.

        Returns:
            A boolean expression, true where the value is in `values`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.select(r=bt.col("x").is_in([1, 3])).to_pydict()
                {'r': [True, False, True]}

                >>> ds = bt.from_pydict({"x": [1, 2, None]})
                >>> ds.select(
                ...     sql=bt.col("x").is_in([1, None]),
                ...     safe=bt.col("x").is_in([1, None], nulls_equal=True),
                ... ).to_pydict()
                {'sql': [True, None, None], 'safe': [True, False, True]}
        """
        vals = list(values)
        if nulls_equal:
            return _null_safe_membership(self, vals)
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
        expr = _membership_test(self, non_null)
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

        if not hasattr(mapping, "items"):
            raise PlanError(
                f"replace(): mapping must be a dict of {{old: new}}, got "
                f"{type(mapping).__name__} {mapping!r}"
            )
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
        return _accessor("batcher.plan.expr_ir.namespaces", "_StrNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.namespaces", "_DtNamespace")(self)

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

    def chr(self) -> Expr:
        """The character at this Unicode code point (DuckDB/Spark ``chr``, → Utf8).

        Returns:
            A new Utf8 expression; null where the value is not a code point.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"n": [65, 233]})
                >>> ds.select(r=bt.col("n").chr()).to_pydict()
                {'r': ['A', 'é']}
        """
        from batcher.plan.expr_ir.func_nodes import StrFunc

        return StrFunc("chr", self)

    def to_base(self, radix: int, *, twos_complement: bool = False) -> Expr:
        """This integer written in `radix` (DuckDB ``to_base``; ``bin`` is radix 2, → Utf8).

        By default a negative value is its magnitude's digits after a ``-``, as DuckDB
        writes it. `twos_complement=True` writes a negative value as the digits of its
        64-bit two's-complement bit pattern instead, which is Spark ``bin`` and ``hex``
        and Daft ``bin``: ``-1`` in radix 2 is sixty-four ``1``s. Only a power-of-two
        radix has such a digit string, so the flag requires one. It is composed from
        existing nodes (the top digit and the low bits rendered separately and padded).

        Args:
            radix: The base, from 2 to 36.
            twos_complement: Render a negative value as its 64-bit two's complement.

        Returns:
            A new Utf8 expression: the digits in uppercase, with a leading ``-`` when
            negative unless `twos_complement` is set.

        Raises:
            PlanError: If `radix` is outside 2..36, or `twos_complement` is set with a
                radix that is not a power of two.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"n": [15, 255]})
                >>> ds.select(b=bt.col("n").to_base(2), h=bt.col("n").to_base(16)).to_pydict()
                {'b': ['1111', '11111111'], 'h': ['F', 'FF']}

                >>> neg = bt.from_pydict({"n": [-1, 5]})
                >>> neg.select(h=bt.col("n").to_base(16, twos_complement=True)).to_pydict()
                {'h': ['FFFFFFFFFFFFFFFF', '5']}
        """
        from batcher.plan.expr_ir.func_nodes import StrFunc

        if not 2 <= radix <= 36:
            raise PlanError(f"to_base(): radix must be between 2 and 36, got {radix}")
        plain = StrFunc("to_base", self, start=radix)
        if not twos_complement:
            return plain
        if radix & (radix - 1):
            raise PlanError(
                f"to_base(twos_complement=True): radix must be a power of two, got {radix}"
            )
        return _twos_complement_digits(self, radix, plain)

    def format_bytes(self, *, si: bool = False) -> Expr:
        """This byte count as human-readable text (DuckDB ``format_bytes``, → Utf8).

        Args:
            si: Use decimal units (``kB``, ``MB``; powers of 1000) instead of the
                default binary ones (``KiB``, ``MiB``; powers of 1024).

        Returns:
            A new Utf8 expression, e.g. ``"1.5 KiB"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"n": [512, 1536]})
                >>> ds.select(r=bt.col("n").format_bytes()).to_pydict()
                {'r': ['512 bytes', '1.5 KiB']}
        """
        from batcher.plan.expr_ir.func_nodes import StrFunc

        return StrFunc("format_bytes_si" if si else "format_bytes", self)

    def round(self, digits: int | None = None, *, mode: str = "half_away_from_zero") -> Expr:
        """Round to the nearest integer, or to `digits` decimal places.

        `mode` picks the tie rule. ``"half_away_from_zero"`` (the default) is DuckDB's
        ``round``: ``2.5`` becomes ``3.0`` and ``-2.5`` becomes ``-3.0``.
        ``"half_to_even"`` is DuckDB's ``round_even``, the Polars and Ray Data default and
        Spark ``bround``: a tie goes to the even neighbour, so ``2.5`` becomes ``2.0``.
        It is also IEEE ``roundTiesToEven``, so summing rounded values does not drift
        upward. An integer input stays an integer under either mode.

        Args:
            digits: Number of decimal places to keep; negative rounds to tens, hundreds
                and so on. ``None`` (the default) rounds to a whole number.
            mode: ``"half_away_from_zero"`` or ``"half_to_even"``.

        Returns:
            A new expression of the rounded values.

        Raises:
            PlanError: If `mode` is not one of the two tie rules.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.234, 2.567]})
                >>> ds.select(r=bt.col("x").round(2)).to_pydict()
                {'r': [1.23, 2.57]}

                >>> ties = bt.from_pydict({"x": [0.5, 1.5, 2.5, -2.5]})
                >>> ties.select(r=bt.col("x").round(mode="half_to_even")).to_pydict()
                {'r': [0.0, 2.0, 2.0, -2.0]}
        """
        if mode == "half_to_even":
            return Math2Expr("round_even", self, Lit(0 if digits is None else digits))
        if mode != "half_away_from_zero":
            raise PlanError(
                f"round(): mode must be 'half_away_from_zero' or 'half_to_even', got {mode!r}"
            )
        if digits is None:
            return MathExpr("round", self)
        return Math2Expr("round", self, Lit(digits))

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

    def square(self) -> Expr:
        """Each value squared, i.e. ``x * x`` (dtype preserved; nulls propagate).

        Returns:
            A new expression of the squared values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [2, 3]})
                >>> ds.select(r=bt.col("x").square()).to_pydict()
                {'r': [4, 9]}
        """
        return self * self

    def log1p(self) -> Expr:
        """Natural log of ``1 + x``, accurate for small ``x`` (→ Float64; nulls propagate).

        The composed ``(1 + x).ln()`` spelling, named for parity with NumPy/DuckDB
        ``log1p``; use it when ``x`` is close to zero and ``ln(1 + x)`` would lose
        precision.

        Returns:
            A new Float64 expression of ``ln(1 + x)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").log1p()).to_pydict()
                {'r': [0.0, 0.6931471805599453]}
        """
        return (Lit(1) + self).ln()

    def expm1(self) -> Expr:
        """``e**x - 1``, accurate for small ``x`` (→ Float64; nulls propagate).

        The inverse of :meth:`log1p`, named for parity with NumPy/DuckDB ``expm1``.

        Returns:
            A new Float64 expression of ``exp(x) - 1``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").expm1()).to_pydict()
                {'r': [0.0, 1.718281828459045]}
        """
        return self.exp() - Lit(1)

    def arcsin(self) -> MathExpr:
        """Arcsine in radians — the Polars/NumPy ``arcsin`` spelling of :meth:`asin`.

        Returns:
            A new Float64 expression of the arcsines, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").arcsin()).to_pydict()
                {'r': [0.0, 1.5707963267948966]}
        """
        return MathExpr("asin", self)

    def arccos(self) -> MathExpr:
        """Arccosine in radians — the Polars/NumPy ``arccos`` spelling of :meth:`acos`.

        Returns:
            A new Float64 expression of the arccosines, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 0.0]})
                >>> ds.select(r=bt.col("x").arccos()).to_pydict()
                {'r': [0.0, 1.5707963267948966]}
        """
        return MathExpr("acos", self)

    def arctan(self) -> MathExpr:
        """Arctangent in radians — the Polars/NumPy ``arctan`` spelling of :meth:`atan`.

        Returns:
            A new Float64 expression of the arctangents, in radians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").arctan()).to_pydict()
                {'r': [0.0, 0.7853981633974483]}
        """
        return MathExpr("atan", self)

    def arcsinh(self) -> Expr:
        """Inverse hyperbolic sine — the Polars/NumPy ``arcsinh`` spelling of :meth:`asinh`.

        Returns:
            A new Float64 expression of the inverse hyperbolic sines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").arcsinh()).to_pydict()
                {'r': [0.0, 0.881373587019543]}
        """
        return MathExpr("asinh", self)

    def arccosh(self) -> Expr:
        """Inverse hyperbolic cosine — the Polars/NumPy ``arccosh`` spelling of :meth:`acosh`.

        Returns:
            A new Float64 expression of the inverse hyperbolic cosines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0]})
                >>> ds.select(r=bt.col("x").arccosh()).to_pydict()
                {'r': [0.0, 1.3169578969248166]}
        """
        return MathExpr("acosh", self)

    def arctanh(self) -> Expr:
        """Inverse hyperbolic tangent — the Polars/NumPy ``arctanh`` spelling of :meth:`atanh`.

        Returns:
            A new Float64 expression of the inverse hyperbolic tangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 0.5]})
                >>> ds.select(r=bt.col("x").arctanh()).to_pydict()
                {'r': [0.0, 0.5493061443340548]}
        """
        return MathExpr("atanh", self)

    # --- feature engineering / ML transforms --------------------------------
    # Scalers broadcast a *window* aggregate over the whole column (or per
    # `partition_by` group) and combine it with the row value, so a fit-and-apply
    # scaling is one pass with no Python state — and, being ordinary window +
    # arithmetic nodes, identical single-node and distributed.

    def _window_mean_std(self, partition_by: Iterable[IntoExpr]) -> tuple[Expr, Expr]:
        """The broadcast ``(mean, sample stddev)`` of this column over its window.

        The window engine offers `sum`/`avg`/`min`/`max`/`count` but no `stddev`, so the
        deviation is built from window aggregates. It uses the **two-pass** form —
        ``E[(x - mean)^2]`` against the already-broadcast window mean — rather than
        ``E[x^2] - E[x]^2``, because the latter subtracts two nearly equal large numbers
        and loses a digit for every digit by which the mean exceeds the spread. On
        ``[k+1, ..., k+6]`` it drove `zscore` to `inf` at ``k=1e9`` (the standard deviation
        cancelled to exactly 0) and to `NaN` at ``k=1e12`` (it cancelled negative, and the
        square root of a negative is not a number). An epoch-second timestamp is ~1.7e9.

        The mean is a window aggregate broadcast to every row, so ``x - mean`` is an
        ordinary scalar expression and the second pass costs one more window aggregate over
        the same partition — which `hoist_windows` shares with the first."""
        keys = list(partition_by)
        n = self.count().over(partition_by=keys).cast("float64")
        mean = self.mean().over(partition_by=keys)
        deviation = self - mean
        var_pop = (deviation * deviation).mean().over(partition_by=keys)
        std = (var_pop * (n / (n - Lit(1)))).sqrt()
        return mean, std

    def zscore(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Standardize to zero mean and unit variance — ``(x - mean) / stddev``.

        The scikit-learn ``StandardScaler`` transform as one expression: the mean and
        sample standard deviation are computed over the whole column, or per group with
        `partition_by`, and broadcast back to every row.

        Args:
            partition_by: Standardize within each group of these key expressions
                instead of across the whole column.

        Returns:
            A Float64 expression of the standardized values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.select(z=bt.col("x").zscore().round(4)).to_pydict()
                {'z': [-1.0, 0.0, 1.0]}
        """
        mean, std = self._window_mean_std(partition_by)
        return (self - mean) / std

    def minmax_scale(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Scale to ``[0, 1]`` — ``(x - min) / (max - min)`` (scikit-learn ``MinMaxScaler``).

        A constant column divides by zero and yields NaN, as the transform is undefined
        there.

        Args:
            partition_by: Scale within each group of these key expressions.

        Returns:
            A Float64 expression of the scaled values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.select(s=bt.col("x").minmax_scale()).to_pydict()
                {'s': [0.0, 0.5, 1.0]}
        """
        keys = list(partition_by)
        lo = self.min().over(partition_by=keys)
        hi = self.max().over(partition_by=keys)
        return (self - lo) / (hi - lo)

    def maxabs_scale(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Scale to ``[-1, 1]`` by the largest magnitude — ``x / max(|x|)``.

        The scikit-learn ``MaxAbsScaler`` transform; it preserves sign and sparsity
        (a zero stays zero) because it never subtracts a centre.

        Args:
            partition_by: Scale within each group of these key expressions.

        Returns:
            A Float64 expression of the scaled values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-1.0, 0.0, 2.0]})
                >>> ds.select(s=bt.col("x").maxabs_scale()).to_pydict()
                {'s': [-0.5, 0.0, 1.0]}
        """
        return self / self.abs().max().over(partition_by=list(partition_by))

    def mean_center(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Subtract the mean — ``x - mean(x)`` — leaving the scale untouched.

        Args:
            partition_by: Centre within each group of these key expressions.

        Returns:
            An expression of the mean-centred values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.select(c=bt.col("x").mean_center()).to_pydict()
                {'c': [-1.0, 0.0, 1.0]}
        """
        return self - self.mean().over(partition_by=list(partition_by))

    def is_outlier(self, threshold: float = 3.0, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """True where the value lies more than `threshold` standard deviations from the mean.

        The z-score outlier rule, as a predicate you can filter on directly.

        Args:
            threshold: How many standard deviations away counts as an outlier.
            partition_by: Judge outliers within each group of these key expressions.

        Returns:
            A Boolean expression, true for the outlying rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 1.0, 1.0, 1.0, 10.0]})
                >>> ds.select(o=bt.col("x").is_outlier(1.5)).to_pydict()
                {'o': [False, False, False, False, True]}
        """
        return self.zscore(partition_by).abs() > Lit(threshold)

    def sigmoid(self) -> Expr:
        """The logistic sigmoid ``1 / (1 + exp(-x))``, mapping any real to ``(0, 1)``.

        The inverse of :meth:`logit`, and the activation that turns a linear score into
        a probability.

        Returns:
            A Float64 expression of the sigmoid values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(p=bt.col("x").sigmoid()).to_pydict()
                {'p': [0.5]}
        """
        return Lit(1.0) / (Lit(1.0) + (Lit(0) - self).exp())

    def logit(self) -> Expr:
        """The log-odds ``ln(x / (1 - x))`` — the inverse of :meth:`sigmoid`.

        Defined for ``0 < x < 1``; the bounds map to ∓inf.

        Returns:
            A Float64 expression of the log-odds.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.5]})
                >>> ds.select(l=bt.col("x").logit()).to_pydict()
                {'l': [0.0]}
        """
        return (self / (Lit(1.0) - self)).ln()

    def relu(self) -> Expr:
        """The rectified linear unit ``max(x, 0)`` — negatives clamped to zero.

        Returns:
            An expression with the negative values replaced by zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-1.0, 0.0, 2.0]})
                >>> ds.select(r=bt.col("x").relu()).to_pydict()
                {'r': [0.0, 0.0, 2.0]}
        """
        return self.clip(lower=Lit(0.0))

    def softplus(self) -> Expr:
        """The smooth rectifier ``ln(1 + exp(x))`` — a differentiable :meth:`relu`.

        Returns:
            A Float64 expression of the softplus values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(s=bt.col("x").softplus()).to_pydict()
                {'s': [0.6931471805599453]}
        """
        return (Lit(1.0) + self.exp()).ln()

    def silu(self) -> Expr:
        """The SiLU / Swish activation ``x * sigmoid(x)``.

        The self-gated activation used across modern architectures (EfficientNet, many
        transformer MLP blocks). Smooth, non-monotonic, and unbounded above.

        Returns:
            A Float64 expression of the SiLU values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(s=bt.col("x").silu()).to_pydict()["s"]]
                [0.0, 0.7311]
        """
        return self * self.sigmoid()

    def gelu(self) -> Expr:
        """The GELU activation (tanh approximation) — the transformer feed-forward default.

        ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``, matching
        ``torch.nn.functional.gelu(x, approximate="tanh")`` (GPT-2 / BERT). Composed from
        the engine's ``tanh`` so it runs in the data plane with no per-row Python.

        Returns:
            A Float64 expression of the GELU values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(g=bt.col("x").gelu()).to_pydict()["g"]]
                [0.0, 0.8412]
        """
        coeff = Lit(math.sqrt(2.0 / math.pi))
        inner = coeff * (self + Lit(0.044715) * self * self * self)
        return Lit(0.5) * self * (Lit(1.0) + inner.tanh())

    def mish(self) -> Expr:
        """The Mish activation ``x * tanh(softplus(x))``.

        A smooth, self-regularizing activation (YOLOv4 and others). Composed from the
        engine's ``softplus`` and ``tanh``. Matches ``torch.nn.functional.mish``.

        Returns:
            A Float64 expression of the Mish values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(m=bt.col("x").mish()).to_pydict()["m"]]
                [0.0, 0.8651]
        """
        return self * self.softplus().tanh()

    def hardsigmoid(self) -> Expr:
        """The hard sigmoid ``clip((x + 3) / 6, 0, 1)`` — the cheap piecewise-linear sigmoid.

        The mobile-friendly approximation used in MobileNetV3. Matches
        ``torch.nn.functional.hardsigmoid``.

        Returns:
            A Float64 expression in ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-3.0, 0.0, 3.0]})
                >>> ds.select(h=bt.col("x").hardsigmoid()).to_pydict()
                {'h': [0.0, 0.5, 1.0]}
        """
        return ((self + Lit(3.0)) / Lit(6.0)).clip(lower=Lit(0.0), upper=Lit(1.0))

    def hardswish(self) -> Expr:
        """The hard swish ``x * hardsigmoid(x)`` — the cheap piecewise-linear SiLU.

        The activation in MobileNetV3's later layers. Matches
        ``torch.nn.functional.hardswish``.

        Returns:
            A Float64 expression of the hard-swish values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-3.0, 0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(h=bt.col("x").hardswish()).to_pydict()["h"]]
                [-0.0, 0.0, 0.6667]
        """
        return self * self.hardsigmoid()

    def leaky_relu(self, negative_slope: float = 0.01) -> Expr:
        """The leaky ReLU: ``x`` for ``x > 0``, else ``negative_slope * x``.

        A ReLU that lets a small gradient through for negative inputs. Matches
        ``torch.nn.functional.leaky_relu``.

        Args:
            negative_slope: The slope applied to negative inputs (default ``0.01``).

        Returns:
            A Float64 expression of the leaky-ReLU values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2.0, 0.0, 3.0]})
                >>> ds.select(r=bt.col("x").leaky_relu()).to_pydict()
                {'r': [-0.02, 0.0, 3.0]}
        """
        from batcher.plan.expr_ir.constructors import when

        return when(self > Lit(0.0)).then(self).otherwise(Lit(negative_slope) * self)

    def elu(self, alpha: float = 1.0) -> Expr:
        """The exponential linear unit: ``x`` for ``x > 0``, else ``alpha * (exp(x) - 1)``.

        A smooth activation with negative saturation at ``-alpha``. Matches
        ``torch.nn.functional.elu``.

        Args:
            alpha: The negative-saturation scale (default ``1.0``).

        Returns:
            A Float64 expression of the ELU values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(r=bt.col("x").elu()).to_pydict()["r"]]
                [0.0, 1.0]
        """
        from batcher.plan.expr_ir.constructors import when

        return when(self > Lit(0.0)).then(self).otherwise(Lit(alpha) * (self.exp() - Lit(1.0)))

    def hardtanh(self) -> Expr:
        """The hard tanh ``clip(x, -1, 1)`` — a cheap piecewise-linear tanh.

        Matches ``torch.nn.functional.hardtanh``.

        Returns:
            A Float64 expression clamped to ``[-1, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2.0, 0.5, 2.0]})
                >>> ds.select(h=bt.col("x").hardtanh()).to_pydict()
                {'h': [-1.0, 0.5, 1.0]}
        """
        return self.clip(lower=Lit(-1.0), upper=Lit(1.0))

    def softsign(self) -> Expr:
        """The softsign activation ``x / (1 + |x|)`` — a smooth, bounded ``(-1, 1)`` map.

        A cheaper-to-compute alternative to tanh. Matches ``torch.nn.functional.softsign``.

        Returns:
            A Float64 expression in ``(-1, 1)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2.0, 0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(s=bt.col("x").softsign()).to_pydict()["s"]]
                [-0.6667, 0.0, 0.5]
        """
        return self / (Lit(1.0) + self.abs())

    def tanhshrink(self) -> Expr:
        """The tanhshrink activation ``x - tanh(x)``.

        Matches ``torch.nn.functional.tanhshrink``.

        Returns:
            A Float64 expression of the tanhshrink values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> [round(v, 4) for v in ds.select(t=bt.col("x").tanhshrink()).to_pydict()["t"]]
                [0.0, 0.2384]
        """
        return self - self.tanh()

    def is_positive(self) -> Expr:
        """True where the value is strictly greater than zero (nulls stay null).

        Returns:
            A Boolean expression, true for positive values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2, 0, 3]})
                >>> ds.select(p=bt.col("x").is_positive()).to_pydict()
                {'p': [False, False, True]}
        """
        return self > Lit(0)

    def is_negative(self) -> Expr:
        """True where the value is strictly less than zero (nulls stay null).

        Returns:
            A Boolean expression, true for negative values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2, 0, 3]})
                >>> ds.select(n=bt.col("x").is_negative()).to_pydict()
                {'n': [True, False, False]}
        """
        return self < Lit(0)

    def is_zero(self) -> Expr:
        """True where the value equals zero (nulls stay null).

        Returns:
            A Boolean expression, true for zero values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2, 0, 3]})
                >>> ds.select(z=bt.col("x").is_zero()).to_pydict()
                {'z': [False, True, False]}
        """
        return self == Lit(0)

    def is_even(self) -> Expr:
        """True where the integer value is divisible by two (nulls stay null).

        Returns:
            A Boolean expression, true for even values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2, 0, 3]})
                >>> ds.select(e=bt.col("x").is_even()).to_pydict()
                {'e': [True, True, False]}
        """
        return self % Lit(2) == Lit(0)

    def is_odd(self) -> Expr:
        """True where the integer value is not divisible by two (nulls stay null).

        Returns:
            A Boolean expression, true for odd values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-2, 0, 3]})
                >>> ds.select(o=bt.col("x").is_odd()).to_pydict()
                {'o': [False, False, True]}
        """
        return self % Lit(2) != Lit(0)

    # --- expanding (cumulative) statistics and encodings ---------------------

    def expanding_mean(
        self,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Running mean of every value up to and including this row (pandas ``expanding().mean()``).

        The cumulative counterpart to :meth:`rolling_mean` — the frame grows instead of
        sliding. Composed as ``cum_sum / cum_count``, so it adds no operator.

        Args:
            partition_by: Restart the running mean per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A Float64 expression of the running mean.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.with_columns(m=bt.col("x").expanding_mean(order_by="x")).to_pydict()["m"]
                [1.0, 1.5, 2.0, 2.5]
        """
        keys, order = list(partition_by), list(order_by)
        total = self.cum_sum(partition_by=keys, order_by=order)
        n = self.cum_count(partition_by=keys, order_by=order)
        return total / n

    def expanding_var(
        self,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        ddof: int = 1,
    ) -> Expr:
        """Running variance of every value up to this row (pandas ``expanding().var()``).

        Built from the running moments ``E[x^2] - E[x]^2`` with the Bessel correction, so
        the first row of each partition (a single value) is undefined and yields NaN.

        Args:
            partition_by: Restart the accumulation per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            ddof: Delta degrees of freedom; ``1`` for sample, ``0`` for population.

        Returns:
            A Float64 expression of the running variance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> v = bt.col("x").expanding_var(order_by="x").round(4)
                >>> ds.with_columns(v=v).to_pydict()["v"]
                [nan, 0.5, 1.0, 1.6667]
        """
        from batcher.plan.expr_ir.constructors import when

        keys, order = list(partition_by), list(order_by)
        # Centered on the partition mean before the running moments are taken. The identity
        # `Var(x) = Var(x - k)` makes this exact for any constant `k`, and without it the
        # `E[x^2] - E[x]^2` difference cancels: on `[k+1, ..., k+6]` the running variance
        # came back as 0.0 at `k=1e9` and as -161061273 -- a negative variance -- at
        # `k=1e12`. The partition mean is the constant nearest the data that a window
        # expression can name; see `_rolling_var`, which carries the same correction.
        centre = AggExpr("avg", self).over(partition_by=keys)
        centered = self - centre
        n = self.cum_count(partition_by=keys, order_by=order).cast("float64")
        mean = centered.cum_sum(partition_by=keys, order_by=order) / n
        mean_sq = (centered * centered).cum_sum(partition_by=keys, order_by=order) / n
        raw = mean_sq - mean * mean
        # Clamped through a comparison rather than a max(), so a NaN from a non-finite
        # value still propagates instead of being reported as a confident zero variance.
        var_pop = when(raw < Lit(0.0)).then(Lit(0.0)).otherwise(raw)
        if ddof == 0:
            return var_pop
        return var_pop * (n / (n - Lit(ddof)))

    def expanding_std(
        self,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        ddof: int = 1,
    ) -> Expr:
        """Running standard deviation up to this row — the square root of :meth:`expanding_var`.

        Args:
            partition_by: Restart the accumulation per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            ddof: Delta degrees of freedom; ``1`` for sample, ``0`` for population.

        Returns:
            A Float64 expression of the running standard deviation.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> s = bt.col("x").expanding_std(order_by="x").round(4)
                >>> ds.with_columns(s=s).to_pydict()["s"]
                [nan, 0.7071, 1.0, 1.291]
        """
        return self.expanding_var(partition_by, order_by, ddof).sqrt()

    def hash_bucket(self, buckets: int, seed: int = 0, *, algorithm: str = "batcher") -> Expr:
        """Assign each value to one of `buckets` by a stable hash — ``|hash(x)| % buckets``.

        Deterministic across partitions, runs, and machines, which is what makes it a
        safe key for a reproducible train/test split, a shard assignment, or an A/B
        bucket.

        `algorithm="iceberg"` computes the Iceberg bucket partition transform instead,
        which is Spark's ``bucket(n, col)``: the Iceberg Murmur3 hash, masked
        non-negative, modulo `buckets`. A null stays null there rather than landing in a
        bucket, and the transform is defined for integers, dates, timestamps, strings,
        binary and decimals only. Use it to read or write a table another engine bucketed.

        Args:
            buckets: How many buckets to spread values across (must be >= 1).
            seed: Hash seed; vary it for an independent bucketing of the same keys. The
                Iceberg transform has no seed, so it must stay 0 there.
            algorithm: ``"batcher"`` (the default) or ``"iceberg"``.

        Returns:
            An Int64 expression in ``[0, buckets)``.

        Raises:
            PlanError: If `buckets` < 1, `algorithm` is unknown, or a seed is given to
                the Iceberg transform.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"k": ["a", "b", "c", "d"]})
                >>> ds.select(b=bt.col("k").hash_bucket(4)).to_pydict()
                {'b': [1, 3, 3, 1]}

                >>> ice = bt.from_pydict({"id": [34, None]})
                >>> ice.select(b=bt.col("id").hash_bucket(16, algorithm="iceberg")).to_pydict()
                {'b': [3, None]}
        """
        buckets = require_int(buckets, func="hash_bucket", arg="buckets", minimum=1)
        seed = require_int(seed, func="hash_bucket", arg="seed")
        if algorithm == "iceberg":
            if seed:
                raise PlanError("hash_bucket(algorithm='iceberg') has no seed; leave it at 0")
            return self.hash(algorithm="iceberg").bitwise_and(Lit(0x7FFFFFFF)) % Lit(buckets)
        if algorithm != "batcher":
            raise PlanError(
                f"hash_bucket(): algorithm must be 'batcher' or 'iceberg', got {algorithm!r}"
            )
        return self.hash(seed=seed).abs() % Lit(buckets)

    def pct_of_total(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Each value as a share of the column total — ``x / sum(x)``, summing to 1.

        The "percent of total" every share/contribution chart needs, computed in one
        pass by broadcasting the windowed sum back over the rows.

        Args:
            partition_by: Take the share within each group of these key expressions.

        Returns:
            A Float64 expression of the per-row share.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.select(p=bt.col("x").pct_of_total()).to_pydict()
                {'p': [0.1, 0.2, 0.3, 0.4]}
        """
        return self / self.sum().over(partition_by=list(partition_by))

    def cumulative_pct(
        self,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Running share of the total — ``cum_sum(x) / sum(x)``, ending at 1.

        The Pareto / cumulative-contribution curve: sort by the value descending and this
        answers "how much of the total do the top N account for".

        Args:
            partition_by: Accumulate within each group of these key expressions.
            order_by: Order rows by these expressions before accumulating.

        Returns:
            A Float64 expression of the running share, rising to 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.with_columns(c=bt.col("x").cumulative_pct(order_by="x")).to_pydict()["c"]
                [0.1, 0.3, 0.6, 1.0]
        """
        keys = list(partition_by)
        running = self.cum_sum(partition_by=keys, order_by=list(order_by))
        # Framed over the whole partition, so an order bound later by `.over(order_by=...)`
        # orders the running numerator without turning the total into a running one too.
        return running / self.sum().over(partition_by=keys, frame=(None, None))

    def normalize_l1(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Scale by the sum of absolute values — ``x / sum(|x|)`` (L1 normalization).

        The signed counterpart to :meth:`pct_of_total`: it handles negative values by
        dividing by the total magnitude, so the absolute shares sum to 1.

        Args:
            partition_by: Normalize within each group of these key expressions.

        Returns:
            A Float64 expression of the L1-normalized values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.select(n=bt.col("x").normalize_l1()).to_pydict()
                {'n': [0.1, 0.2, 0.3, 0.4]}
        """
        return self / self.abs().sum().over(partition_by=list(partition_by))

    def safe_divide(self, other: IntoExpr) -> Expr:
        """Divide, yielding null instead of an error or infinity when `other` is zero.

        ``x / 0`` is the classic silent-corruption source in a derived metric; this makes
        the undefined rows explicitly null so they propagate and can be filtered.

        Args:
            other: The divisor; rows where it is zero produce null.

        Returns:
            A Float64 expression of the quotient, null where the divisor is zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [2.0, 0.0]})
                >>> ds.select(r=bt.col("a").safe_divide(bt.col("b"))).to_pydict()
                {'r': [0.5, None]}
        """
        from batcher.plan.expr_ir.constructors import nullif

        divisor = _wrap(other)
        return self / nullif(divisor, Lit(0))

    def rank_pct(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Percentile rank of each value in ``[0, 1]``, ascending — SQL ``PERCENT_RANK``.

        The distribution-free position of a value among its peers: 0 for the smallest,
        1 for the largest. Useful as a scale-free feature when the raw magnitude varies
        between groups or over time. For a descending rank, rank the negated value.

        Args:
            partition_by: Rank within each group of these key expressions.

        Returns:
            A Float64 expression of the percentile rank.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.with_columns(p=bt.col("x").rank_pct()).to_pydict()["p"]
                [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
        """
        from batcher.plan.expr_ir.nodes import percent_rank

        return percent_rank().over(partition_by=list(partition_by), order_by=[self])

    def softmax(self, partition_by: Iterable[IntoExpr] = ()) -> Expr:
        """Softmax over the column — ``exp(x) / sum(exp(x))``, a distribution summing to 1.

        Turns a column of scores into probabilities. Computed by broadcasting the
        windowed sum of the exponentials, so it is one pass with no Python state.

        Args:
            partition_by: Normalize within each group of these key expressions.

        Returns:
            A Float64 expression of the softmax probabilities.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0]})
                >>> ds.select(p=bt.col("x").softmax().round(4)).to_pydict()
                {'p': [0.09, 0.2447, 0.6652]}
        """
        weights = self.exp()
        return weights / weights.sum().over(partition_by=list(partition_by))

    def abs_diff(self, other: IntoExpr) -> Expr:
        """Absolute difference from `other` — ``|x - other|``.

        The unsigned error/distance every comparison and drift check needs.

        Args:
            other: The value or expression to compare against.

        Returns:
            An expression of the absolute difference.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1.0, 5.0], "b": [3.0, 2.0]})
                >>> ds.select(d=bt.col("a").abs_diff(bt.col("b"))).to_pydict()
                {'d': [2.0, 3.0]}
        """
        return (self - _wrap(other)).abs()

    def is_first_distinct(self, order_by: IntoExpr | None = None) -> Expr:
        """True on the first occurrence of each distinct value, in `order_by` order.

        The de-duplication marker: filtering on it keeps one row per distinct value.
        An order is required so the choice is deterministic and partition-independent
        (an arrival-order "first" would differ between a single-node and a distributed
        run); give it here or through ``.over(order_by=...)``.

        Args:
            order_by: The expression whose ascending order decides which occurrence
                counts as first. Omit it only when ``.over(order_by=...)`` supplies it.

        Returns:
            A Boolean expression, true on each value's first row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"i": [0, 1, 2], "c": ["a", "b", "a"]})
                >>> ds.select(f=bt.col("c").is_first_distinct(bt.col("i"))).to_pydict()
                {'f': [True, True, False]}
        """
        from batcher.plan.expr_ir.nodes import row_number

        order = [] if order_by is None else [_col_or_expr(order_by)]
        rn = row_number().over(partition_by=[self], order_by=order)
        return rn == Lit(1)

    def is_last_distinct(self, order_by: IntoExpr | None = None) -> Expr:
        """True on the last occurrence of each distinct value, in `order_by` order.

        The mirror of :meth:`is_first_distinct`, useful for keeping the most recent row
        per key. An order is likewise required for determinism, here or through
        ``.over(order_by=...)``.

        Args:
            order_by: The expression whose ascending order decides which occurrence
                counts as last. Omit it only when ``.over(order_by=...)`` supplies it.

        Returns:
            A Boolean expression, true on each value's last row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"i": [0, 1, 2], "c": ["a", "b", "a"]})
                >>> ds.select(l=bt.col("c").is_last_distinct(bt.col("i"))).to_pydict()
                {'l': [False, True, True]}
        """
        from batcher.plan.expr_ir.nodes import row_number

        order = [] if order_by is None else [_col_or_expr(order_by)]
        rn = row_number().over(partition_by=[self], order_by=order)
        # Framed over the whole partition, so an order bound later by `.over(order_by=...)`
        # cannot turn the group size into a running count.
        total = AggExpr("count", Lit(1)).over(partition_by=[self], frame=(None, None))
        return rn == total

    def label_encode(self) -> Expr:
        """Map each distinct value to a 0-based integer code, ordered by value.

        The scikit-learn ``LabelEncoder`` transform as one expression: the codes are
        assigned by sorting the distinct values, so they are deterministic and identical
        single-node and distributed (an arrival-order encoding would not be).

        Returns:
            An Int64 expression of the 0-based codes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["b", "a", "b", "c"]})
                >>> ds.select(code=bt.col("c").label_encode()).to_pydict()
                {'code': [1, 0, 1, 2]}
        """
        from batcher.plan.expr_ir.nodes import dense_rank

        return dense_rank().over(order_by=[self]) - Lit(1)

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

    def sec(self) -> MathExpr:
        """Secant (``1 / cos``) of an angle in radians (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the secants.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0]})
                >>> ds.select(r=bt.col("x").sec()).to_pydict()
                {'r': [1.0]}
        """
        return MathExpr("sec", self)

    def csc(self) -> MathExpr:
        """Cosecant (``1 / sin``) of an angle in radians (→ Float64; nulls propagate).

        Returns:
            A new Float64 expression of the cosecants.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import math
                >>> ds = bt.from_pydict({"x": [math.pi / 2]})
                >>> ds.select(r=bt.col("x").csc()).to_pydict()
                {'r': [1.0]}
        """
        return MathExpr("csc", self)

    def even(self) -> MathExpr:
        """Round away from zero to the nearest even integer (DuckDB ``even``; → Float64).

        The rounding direction is *outward*, not to-nearest: ``3.0`` becomes ``4.0`` and
        ``-2.1`` becomes ``-4.0``. A value that is already an even integer is unchanged.

        Returns:
            A new Float64 expression of the rounded values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [2.1, -2.1, 2.0, 3.0]})
                >>> ds.select(r=bt.col("x").even()).to_pydict()
                {'r': [4.0, -4.0, 2.0, 4.0]}
        """
        return MathExpr("even", self)

    def gamma(self) -> MathExpr:
        """The gamma function ``Γ(x)`` (DuckDB ``gamma``; → Float64).

        The continuous extension of the factorial: ``Γ(n) == (n - 1)!`` for a positive
        integer. Use :meth:`lgamma` instead above ~171, where ``Γ`` overflows to infinity.

        Returns:
            A new Float64 expression of the gamma values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5.0, 1.0]})
                >>> ds.select(r=bt.col("x").gamma()).to_pydict()
                {'r': [24.0, 1.0]}
        """
        return MathExpr("gamma", self)

    def lgamma(self) -> MathExpr:
        """The natural log of ``|Γ(x)|`` (DuckDB ``lgamma``; → Float64).

        Computed directly rather than as ``gamma().ln()``, which overflows to infinity
        above ~171 and loses the answer entirely. This is the form log-likelihoods and
        combinatorial ratios are written in.

        Returns:
            A new Float64 expression of the log-gamma values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5.0]})
                >>> r = ds.select(r=bt.col("x").lgamma()).to_pydict()
                >>> round(r["r"][0], 6)
                3.178054
        """
        return MathExpr("lgamma", self)

    def factorial(self) -> MathExpr:
        """``n!`` — factorial of a non-negative integer (DuckDB ``factorial``; → Int64).

        Computed exactly in 64-bit integers, so the defined inputs are ``0`` through
        ``20``; a negative input or one past ``20!`` raises rather than wrapping. Spark
        ``factorial`` answers null outside that range instead, which
        ``bt.when(n.between(0, 20)).then(n.clip(0, 20).factorial()).otherwise(bt.lit(None))``
        reproduces.

        Returns:
            A new Int64 expression of the factorials.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5]})
                >>> ds.select(f=bt.col("x").factorial()).to_pydict()
                {'f': [120]}
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
                {'r': [3]}
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
        return _accessor("batcher.plan.expr_ir.namespaces", "_ListNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.namespaces", "_StructNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.namespaces", "_MapNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.namespaces", "_JsonNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.image", "_ImageNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.audio", "_AudioNamespace")(self)

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
        return _accessor("batcher.plan.expr_ir.video", "_VideoNamespace")(self)

    @property
    def seq(self) -> _SeqNamespace:
        """Sequence accessor — genomics and proteomics ops on a text column.

        Returns a namespace with ops such as ``.seq.reverse_complement()``,
        ``.seq.gc_content()``, ``.seq.translate()``, ``.seq.kmers(21)``, and the FASTQ
        quality decoders; all per-base work stays in the Rust data plane.

        Returns:
            The `.seq` accessor namespace.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.col("s").seq.reverse_complement().to_ir()["e"]
                'seq'
        """
        return _accessor("batcher.plan.expr_ir.namespaces.sequence", "_SeqNamespace")(self)

    def hash(self, seed: int = 0, *, algorithm: str = "batcher") -> Expr:
        """A deterministic 64-bit hash of this expression's value, per row → Int64.

        The single-argument spelling of :func:`batcher.hash_rows`. Typed rather than
        textual, so it neither depends on how a float renders nor pays to render it.
        `algorithm` reproduces another engine's hash instead; see
        :func:`batcher.hash_rows` for the three choices.

        Args:
            seed: Changes the digest; the same seed reproduces it.
            algorithm: ``"batcher"`` (the default), ``"murmur3"`` (Spark ``hash``),
                ``"iceberg"`` (the Iceberg bucket hash) or ``"xxhash3"`` (Daft ``hash``).

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

        return hash_rows(self, seed=seed, algorithm=algorithm)

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
    def sum(self, *, empty_value: int | float | None = None) -> AggExpr | Expr:
        """Sum of non-null values per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        An aggregate: it collapses a group to one row (or, via :meth:`AggExpr.over`,
        broadcasts the group result to each row). Mergeable, so identical single-node
        and distributed.

        A group with no non-null value sums to null, as in SQL. ``empty_value=0`` answers
        ``0`` there instead, which is what Polars' ``sum`` returns. That form is an
        expression over the aggregate, so use it in ``agg(...)`` rather than ``.over(...)``.

        Args:
            empty_value: The result for a group with no non-null value; ``None`` keeps SQL's
                null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(total=bt.col("x").sum()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'total': [3, 10]}

                >>> nulls = bt.from_pydict({"x": [None, None]})
                >>> nulls = nulls.with_columns(x=bt.col("x").cast("int64"))
                >>> x = bt.col("x")
                >>> nulls.agg(sql=x.sum(), polars=x.sum(empty_value=0)).to_pydict()
                {'sql': [None], 'polars': [0]}
        """
        agg = AggExpr("sum", self)
        if empty_value is None:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.with_empty_value(agg, empty_value)

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

    def max(self, *, nan_policy: str = "propagate") -> AggExpr | Expr:
        """Maximum non-null value per group. Use in ``group_by().agg(...)`` or ``.over(...)``.

        Floats follow SQL's total order, in which NaN is greater than every number, so one
        NaN in a group is its maximum (``nan_policy="propagate"``, DuckDB's answer).
        ``nan_policy="ignore"`` skips NaN and answers NaN only for a group holding nothing
        else, which is Polars' ``max``. That form is an expression over aggregates, so use
        it in ``agg(...)`` rather than ``.over(...)``.

        Args:
            nan_policy: ``"propagate"`` (NaN is the greatest value) or ``"ignore"`` (NaN is
                skipped unless every value is NaN).

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Raises:
            PlanError: If `nan_policy` is not one of the two policies.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").max()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}

                >>> nan = bt.from_pydict({"x": [1.0, float("nan"), 3.0]})
                >>> x = bt.col("x")
                >>> nan.agg(sql=x.max(), polars=x.max(nan_policy="ignore")).to_pydict()
                {'sql': [nan], 'polars': [3.0]}
        """
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.nan_ignoring_max(self, nan_policy)

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

    def var(self, *, ddof: int = 1) -> AggExpr | Expr:
        """Sample variance per group, Bessel-corrected (divides by ``n - 1``).

        An aggregate for ``group_by().agg(...)``. ``ddof`` sets the divisor to ``n - ddof``
        as Polars and Ray Data do: ``ddof=0`` is the population variance. A group with
        ``n <= ddof`` values is null.

        Args:
            ddof: Delta degrees of freedom; the sum of squared deviations is divided by
                ``n - ddof``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 4, 6]})
                >>> ds.group_by("g").agg(r=bt.col("x").var()).to_pydict()
                {'g': ['a'], 'r': [4.0]}
                >>> ds.group_by("g").agg(r=bt.col("x").var(ddof=0).round(4)).to_pydict()
                {'g': ['a'], 'r': [2.6667]}
        """
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.variance_ddof(self, ddof, sqrt=False)

    def std(self, *, ddof: int = 1) -> AggExpr | Expr:
        """Sample standard deviation per group — the square root of :meth:`var`.

        An aggregate for ``group_by().agg(...)``. ``ddof`` sets the variance divisor to
        ``n - ddof``, as Polars' and Ray Data's ``std(ddof=...)`` do. A group with
        ``n <= ddof`` values is null; Ray Data answers NaN there.

        Args:
            ddof: Delta degrees of freedom; the sum of squared deviations is divided by
                ``n - ddof`` before the square root.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [2, 4, 6]})
                >>> ds.group_by("g").agg(r=bt.col("x").std()).to_pydict()
                {'g': ['a'], 'r': [2.0]}
                >>> ds.group_by("g").agg(r=bt.col("x").std(ddof=0).round(4)).to_pydict()
                {'g': ['a'], 'r': [1.633]}
        """
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.variance_ddof(self, ddof, sqrt=True)

    def skew(self, *, bias: bool = False) -> AggExpr:
        """Sample skewness per group (adjusted Fisher-Pearson, matching DuckDB; → Float64).

        Null when the group has fewer than 3 values. Mergeable (sum-of-powers moment state).

        ``bias=True`` is the population skewness ``m3 / m2^1.5`` instead, the default of
        Spark's ``skewness``, Polars' ``skew`` and Daft's ``skew``. It is defined from one
        value up and is null for a group with no variance, where Polars answers NaN.

        Args:
            bias: Whether to return the biased population estimate rather than the
                sample-adjusted one.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").skew()).to_pydict()
                {'g': ['a'], 'r': [1.763632614803888]}
                >>> ds.group_by("g").agg(r=bt.col("x").skew(bias=True).round(6)).to_pydict()
                {'g': ['a'], 'r': [1.018233]}
        """
        return AggExpr("skewness_pop" if bias else "skewness", self)

    def kurtosis(self, *, bias: bool = False, fisher: bool = True) -> AggExpr | Expr:
        """Sample excess kurtosis per group (0 for a normal distribution; → Float64).

        Matches DuckDB. Null when the group has fewer than 4 values. Mergeable.

        ``bias=True`` is the population excess kurtosis ``m4 / m2² - 3`` (Spark's
        ``kurtosis``, Polars' default, and DuckDB's ``kurtosis_pop``).
        ``fisher=False`` reports Pearson's kurtosis, which is the excess plus 3.

        Args:
            bias: Whether to return the biased population estimate rather than the
                sample-corrected one.
            fisher: Whether to subtract 3, so a normal distribution scores 0.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").kurtosis()).to_pydict()
                {'g': ['a'], 'r': [3.152000000000001]}
                >>> pearson = bt.col("x").kurtosis(bias=True, fisher=False)
                >>> ds.group_by("g").agg(r=pearson).to_pydict()
                {'g': ['a'], 'r': [2.7880000000000003]}
        """
        agg = AggExpr("kurtosis_pop" if bias else "kurtosis", self)
        return agg if fisher else agg + Lit(3.0)

    def entropy(
        self, base: float = 2.0, *, of: str = "frequencies", normalize: bool = True
    ) -> AggExpr | Expr:
        """Base-2 Shannon entropy of a group's value distribution (→ Float64).

        ``-Σ pᵢ·log₂(pᵢ)`` over the distinct values' frequencies: 0 when a group holds
        one distinct value, ``log₂(n)`` when all n are distinct. The measure to reach
        for when the question is how *concentrated* a column is, rather than how large.
        This is DuckDB's ``entropy``.

        ``of="values"`` reads the column itself as the probabilities instead, which is
        Polars' ``entropy``: with `normalize` they are first scaled to sum to 1, and without
        it they are used as given. Polars' default base is ``math.e``. A zero or negative
        value is NaN there, as in Polars. Every form other than the default is an
        expression over aggregates, so use it in ``agg(...)``.

        Args:
            base: The logarithm base; 2 measures bits, ``math.e`` nats.
            of: ``"frequencies"`` (how often each distinct value occurs) or ``"values"``
                (the values are the probabilities).
            normalize: With ``of="values"``, whether to divide the values by their sum.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 1, 2, 2]})
                >>> ds.group_by("g").agg(r=bt.col("x").entropy()).to_pydict()
                {'g': ['a'], 'r': [1.0]}

                >>> p = bt.from_pydict({"p": [1.0, 2.0, 3.0]})
                >>> p.agg(h=bt.col("p").entropy(base=2, of="values").round(6)).to_pydict()
                {'h': [1.459148]}
        """
        if base == 2.0 and of == "frequencies" and normalize:
            return AggExpr("entropy", self)
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.entropy_of(self, base, of, normalize)

    def mad(self) -> AggExpr:
        """Median absolute deviation per group (→ Float64).

        ``median(|x - median(x)|)`` — a spread measure that, unlike the standard
        deviation, a single extreme value cannot move.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 100]})
                >>> ds.group_by("g").agg(r=bt.col("x").mad()).to_pydict()
                {'g': ['a'], 'r': [1.0]}
        """
        return AggExpr("mad", self)

    def quantile_disc(self, q: float) -> AggExpr:
        """Discrete quantile `q ∈ [0, 1]` — a value that is actually present (→ Float64).

        Where :meth:`quantile` interpolates between the two bracketing values, this
        returns the element at rank ``ceil(q·n) - 1``. That matters for an ordinal
        column, where the interpolated value may not be a legal value at all.

        Args:
            q: The quantile in ``[0, 1]``.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 2, 3, 4]})
                >>> ds.group_by("g").agg(r=bt.col("x").quantile_disc(0.5)).to_pydict()
                {'g': ['a'], 'r': [2.0]}
        """
        q = require_float(q, func="quantile_disc", arg="q")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile_disc q must be in [0, 1], got {q}")
        return AggExpr("quantile_disc", self, param=q)

    def mode_top_k(self, k: int) -> AggExpr:
        """The `k` most frequent values per group, most frequent first (→ List).

        DuckDB's ``approx_top_k``, computed **exactly**: the aggregate already holds
        every value of the group, so a sketch could only lose accuracy. Ties break to
        the smaller value, so the result does not depend on partition order. The `k`
        *largest* values are :meth:`top_k`.

        Args:
            k: How many values to return.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 6, "x": [1, 2, 2, 3, 3, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").mode_top_k(2)).to_pydict()
                {'g': ['a'], 'r': [[3, 2]]}
        """
        return AggExpr("approx_top_k", self, param=float(k))

    def top_k(self, k: int) -> Expr:
        """The `k` largest non-null values per group, largest first (Polars ``top_k``, → List).

        Composed as the group's collected values with the nulls dropped, sorted
        descending and cut to `k`, so it holds the whole group in memory the way
        :meth:`array_agg` does, and is mergeable for the same reason. Ties keep every
        copy (``[5, 5]``). Nulls are not values here, as in DuckDB's ``max(x, k)``, so a
        group with fewer than `k` non-null values returns what it has, where Polars pads
        the list with the group's nulls. The `k` most *frequent* values are
        :meth:`mode_top_k`.

        Args:
            k: How many values to return (must be >= 1).

        Returns:
            An expression over an aggregate, for use in ``agg(...)``.

        Raises:
            PlanError: If `k` < 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 5, None, 3, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").top_k(2)).to_pydict()
                {'g': ['a'], 'r': [[5, 5]]}
        """
        from batcher.plan.expr_ir.func_nodes import ListFunc, ListSlice
        from batcher.plan.expr_ir.namespaces.collections import _ListNamespace

        k = require_int(k, func="top_k", arg="k", minimum=1)
        values = _ListNamespace(self.array_agg())  # type: ignore[arg-type]
        return ListSlice(ListFunc("sort_desc", values.drop_nulls()), 0, k)

    def kahan_sum(self) -> AggExpr:
        """Compensated sum of a group's values (DuckDB ``fsum``/``kahan_sum``, → Float64).

        A plain float sum loses the low bits of every addend far smaller than the running
        total, so a long column of small values added to a large one drifts. This one
        carries that lost part along and adds it back, which is exact where it matters and
        never worse than :meth:`sum`. Mergeable, so a distributed run agrees.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1e16, 1.0, 1.0, -1e16]})
                >>> ds.select(exact=bt.col("x").kahan_sum(), naive=bt.col("x").sum()).to_pydict()
                {'exact': [2.0], 'naive': [0.0]}
        """
        return AggExpr("kahan_sum", self)

    def any_value(self) -> AggExpr:
        """One value from each group, unspecified which (→ the input type).

        DuckDB's ``any_value``/``arbitrary``, for the common case of carrying a column
        that is constant within the group through a ``group_by`` without naming a
        reduction for it. The engine resolves "unspecified" to the group's **minimum**,
        because a mergeable aggregate has to combine commutatively — so the answer is
        the same on one node as on a hundred, which "the first row" would not be.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "dept": ["eng", "eng"]})
                >>> ds.group_by("g").agg(r=bt.col("dept").any_value()).to_pydict()
                {'g': ['a'], 'r': ['eng']}
        """
        return AggExpr("any_value", self)

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

    def quantile(self, q: float, interpolation: str = "linear") -> AggExpr:
        """Continuous quantile at ``q`` in [0, 1] (linear interpolation).

        ``quantile(0.5)`` equals :meth:`median`. Raises ``PlanError`` if ``q`` is
        outside [0, 1].

        `interpolation` decides what happens when rank ``q·(n-1)`` falls between two
        values, with Polars' names: ``"linear"`` (DuckDB's ``quantile_cont``, the default),
        ``"lower"``, ``"higher"``, ``"nearest"`` (half rounds away from zero, Polars'
        default), ``"midpoint"``, and ``"equiprobable"`` (the element at rank
        ``ceil(q·n) - 1``, which is :meth:`quantile_disc`).

        Args:
            q: The quantile in ``[0, 1]``.
            interpolation: How to resolve a rank between two values.

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
                >>> nearest = bt.col("x").quantile(0.5, "nearest")
                >>> ds.group_by("g").agg(r=nearest).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2.0, 10.0]}
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.ir_tags import QUANTILE_INTERPOLATIONS

        q = require_float(q, func="quantile", arg="q")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile q must be in [0, 1], got {q}")
        if interpolation == "equiprobable":
            return AggExpr("quantile_disc", self, param=q)
        if interpolation not in QUANTILE_INTERPOLATIONS:
            raise PlanError(
                "quantile interpolation must be one of "
                f"{sorted(QUANTILE_INTERPOLATIONS | {'equiprobable'})}, got {interpolation!r}"
            )
        mode = None if interpolation == "linear" else interpolation
        return AggExpr("quantile", self, param=q, interpolation=mode)

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

    def count_distinct(self, *, count_nulls: bool = False) -> AggExpr | Expr:
        """Number of distinct non-null values per group (SQL ``COUNT(DISTINCT)``).

        Exact, so it holds every distinct value — see :meth:`approx_count_distinct` for the
        bounded-memory, skew-safe sketch. An aggregate for ``group_by().agg(...)``.

        ``count_nulls=True`` counts null as one more distinct value when the group has
        one, as Polars' ``n_unique`` does.

        Args:
            count_nulls: Whether a null counts as a distinct value.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").count_distinct()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 1]}

                >>> nulls = bt.from_pydict({"x": [1, None, 1]})
                >>> nulls.agg(n=bt.col("x").count_distinct(count_nulls=True)).to_pydict()
                {'n': [2]}
        """
        agg = AggExpr("count_distinct", self)
        if not count_nulls:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.count_nulls_as_value(agg, self)

    def approx_count_distinct(self, *, count_nulls: bool = False) -> AggExpr | Expr:
        """Approximate COUNT(DISTINCT) via a HyperLogLog sketch (~2% error).

        Bounded memory regardless of skew — the skew-safe choice when an exact
        `count_distinct` on a hot key would hold every distinct value. Mergeable, so it
        is identical single-node and distributed.

        ``count_nulls=True`` adds one for a group holding a null, as Polars'
        ``approx_n_unique`` does. The null is counted exactly, not estimated.

        Args:
            count_nulls: Whether a null counts as a distinct value.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [1, 2, 3]})
                >>> ds.group_by("g").agg(n=bt.col("x").approx_count_distinct()).to_pydict()
                {'g': ['a'], 'n': [3]}
        """
        agg = AggExpr("approx_count_distinct", self)
        if not count_nulls:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.count_nulls_as_value(agg, self)

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
        q = require_float(q, func="approx_quantile", arg="q")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"approx_quantile(q) requires q in [0, 1], got {q}")
        return AggExpr("approx_quantile", self, param=q)

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

    def mode(self, *, all_modes: bool = False) -> AggExpr:
        """Most frequent value per group, ties broken by the smallest value.

        Deterministic and partition-independent. Works on any column type.

        ``all_modes=True`` returns **every** most-frequent value as a list, ascending,
        which is what Polars' ``mode`` returns. Nulls are not counted in either form.

        Args:
            all_modes: Whether to return every tied value as a list rather than one value.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").mode()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}

                >>> tied = bt.from_pydict({"x": [3, 3, 2, 2, 1]})
                >>> tied.agg(r=bt.col("x").mode(all_modes=True)).to_pydict()
                {'r': [[2, 3]]}
        """
        return AggExpr("modes" if all_modes else "mode", self)

    def n50(self) -> AggExpr:
        """Assembly N50 — the contig length at which half the assembly's bases are covered.

        Sort the pieces longest-first and walk down: N50 is the length of the piece at which
        the running total first reaches half the total. It is the standard measure of how
        contiguous an assembly is.

        **This is not the median of the same lengths, and the difference is the point.** A
        median weighs every contig equally, so an assembly of one 10 Mb chromosome plus a
        thousand 500 bp fragments has a median of 500 — a number describing the debris. N50
        weighs by *base* and answers 10 Mb.

        Mergeable, so a per-sample N50 over a shuffle is the same number a single node would
        compute. Null lengths, negative lengths, and non-finite values are excluded.

        Returns:
            An aggregate expression yielding Float64; null for a group with no usable
            lengths, or whose lengths sum to zero.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"asm": ["a"] * 9, "len": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
                >>> ds.group_by("asm").agg(n=bt.col("len").n50()).to_pydict()
                {'asm': ['a'], 'n': [7.0]}
        """
        return AggExpr("n_length", self, param=0.5)

    def n90(self) -> AggExpr:
        """Assembly N90 — the contig length covering 90% of the assembly's bases.

        The same walk as :meth:`n50` at a stricter threshold, so it reaches further down the
        length distribution and is never larger than N50. Where N50 says how big the good
        pieces are, N90 says how far the assembly stays good.

        Returns:
            An aggregate expression yielding Float64; null on the same conditions as
            :meth:`n50`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"asm": ["a"] * 9, "len": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
                >>> ds.group_by("asm").agg(n=bt.col("len").n90()).to_pydict()
                {'asm': ['a'], 'n': [3.0]}
        """
        return AggExpr("n_length", self, param=0.9)

    def l50(self) -> AggExpr:
        """Assembly L50 — how many contigs are needed to cover half the bases.

        The companion of :meth:`n50` and the one it is confused with: **N is a length, L is a
        count**. A low L50 means a few big pieces carry the assembly.

        Returns:
            An aggregate expression yielding Int64; null on the same conditions as
            :meth:`n50`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"asm": ["a"] * 9, "len": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
                >>> ds.group_by("asm").agg(l=bt.col("len").l50()).to_pydict()
                {'asm': ['a'], 'l': [3]}
        """
        return AggExpr("l_count", self, param=0.5)

    def aun(self) -> AggExpr:
        """Area under the Nx curve — the threshold-free contiguity statistic, ``sum(l²)/sum(l)``.

        Equivalently the base-weighted mean length: every base contributes the length of the
        contig holding it. It exists because N50 is a *step* function of the length
        distribution — one contig crossing the halfway mark moves it discontinuously, so two
        assemblies can swap rank on a rounding difference. auN integrates over every threshold
        instead and is continuous in the lengths, which makes it the better number to rank on.

        Needs no sort, so it is the cheapest of the four as well.

        Returns:
            An aggregate expression yielding Float64; null on the same conditions as
            :meth:`n50`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"asm": ["a"] * 3, "len": [10, 20, 30]})
                >>> round(ds.group_by("asm").agg(a=bt.col("len").aun()).to_pydict()["a"][0], 4)
                23.3333
        """
        return AggExpr("aun", self)

    def first(self, order_by: IntoExpr | None = None, *, ignore_nulls: bool = True) -> AggExpr:
        """This expression's value at the first row in `order_by` order (SQL ``first``).

        Equivalent to ``arg_min(order_by)``, and that equivalence is the precise
        contract: like ``arg_min``, this **skips rows where the expression is null** and
        returns the first non-null value. SQL's own ``FIRST(x ORDER BY k)`` does not --
        it returns whatever sits in the first row, null included -- so the two agree on
        every column without nulls and differ exactly where one has them.
        ``ignore_nulls=False`` is that SQL form, and also Spark's ``first`` and Polars'
        ``first``: the value of the first row, even when it is null.

        An order is **required**: an arrival-order first/last is not
        partition-independent, so it could not stay identical single-node and
        distributed. Give it here, or through ``.over(order_by=...)`` when the first value
        is taken per window. With an order key the result is deterministic and mergeable
        (ties on the key break to the smallest value).

        Args:
            order_by: The ordering expression; the value at its first row is returned.
                Omit it only when an enclosing ``.over(order_by=...)`` supplies the order.
            ignore_nulls: Whether to skip rows whose value is null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").first(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}

                >>> held = bt.from_pydict({"x": [None, 2], "t": [1, 2]})
                >>> held.agg(r=bt.col("x").first("t", ignore_nulls=False)).to_pydict()
                {'r': [None]}
                >>> w = bt.col("x").first().over("g", order_by="t")
                >>> ds.with_columns(f=w).sort("g", "t").to_pydict()["f"]
                [2, 2, 10]
        """
        func = "arg_min" if ignore_nulls else "arg_min_null"
        by = None if order_by is None else _col_or_expr(order_by)
        return AggExpr(func, self, input2=by)

    def last(self, order_by: IntoExpr | None = None, *, ignore_nulls: bool = True) -> AggExpr:
        """This expression's value at the last row in `order_by` order (SQL ``last``).

        Equivalent to ``arg_max(order_by)``, which -- as on :meth:`first` -- means it
        **skips nulls** where SQL's ``LAST(x ORDER BY k)`` would return one;
        ``ignore_nulls=False`` returns it. As with :meth:`first`, an explicit `order_by`
        is required so the result stays deterministic and mergeable across partitions.

        Args:
            order_by: The ordering expression; the value at its last row is returned.
                Omit it only when an enclosing ``.over(order_by=...)`` supplies the order.
            ignore_nulls: Whether to skip rows whose value is null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").last(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}
        """
        func = "arg_max" if ignore_nulls else "arg_max_null"
        by = None if order_by is None else _col_or_expr(order_by)
        return AggExpr(func, self, input2=by)

    def min_by(self, by: IntoExpr, *, ignore_nulls: bool = True) -> AggExpr:
        """This expression's value at the row where `by` is minimal (SQL ``min_by``/``arg_min``).

        Key ties break to the smallest value, so the result is deterministic and
        partition-independent. A row whose `by` is null never counts. By default a row
        whose value is null does not either (DuckDB ``arg_min``); ``ignore_nulls=False``
        lets it win and return its null (DuckDB ``arg_min_null``, Spark and Polars
        ``min_by``). The *position* of the minimum is :meth:`arg_min`.

        Args:
            by: The expression whose minimum selects the row; a bare string names a column.
            ignore_nulls: Whether to skip rows whose value is null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").min_by(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [2, 10]}
        """
        func = "arg_min" if ignore_nulls else "arg_min_null"
        return AggExpr(func, self, input2=_col_or_expr(by))

    def max_by(self, by: IntoExpr, *, ignore_nulls: bool = True) -> AggExpr:
        """This expression's value at the row where `by` is maximal (SQL ``max_by``/``arg_max``).

        The *position* of the maximum is :meth:`arg_max`.

        Null handling mirrors :meth:`arg_min`: ``ignore_nulls=False`` is DuckDB's
        ``arg_max_null`` and Spark's and Polars' ``max_by``.

        Args:
            by: The expression whose maximum selects the row; a bare string names a column.
            ignore_nulls: Whether to skip rows whose value is null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").max_by(bt.col("t"))).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [1, 10]}

                >>> held = bt.from_pydict({"x": [1, None], "t": [1, 2]})
                >>> held.agg(r=bt.col("x").max_by("t", ignore_nulls=False)).to_pydict()
                {'r': [None]}
        """
        func = "arg_max" if ignore_nulls else "arg_max_null"
        return AggExpr(func, self, input2=_col_or_expr(by))

    def arg_min(self) -> Expr:
        """The 0-based position of the group's smallest non-null value (Polars ``arg_min``).

        The first position wins a tie, and a group with no non-null value is null.
        Positions count the group's rows in arrival order, nulls included, so the answer
        is only as defined as that order: sort first when it matters. Composed as
        :meth:`array_agg` followed by the list's own ``arg_min``. The *value* at another
        column's minimum is :meth:`min_by`.

        Returns:
            An Int64 expression over an aggregate, for use in ``agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [4, None, 1, 1]})
                >>> ds.agg(r=bt.col("x").arg_min()).to_pydict()
                {'r': [2]}
        """
        from batcher.plan.expr_ir.func_nodes import ListFunc

        return ListFunc("arg_min", self.array_agg())

    def arg_max(self) -> Expr:
        """The 0-based position of the group's largest non-null value (Polars ``arg_max``).

        The first position wins a tie, and a group with no non-null value is null.
        Positions count the group's rows in arrival order, nulls included, so the answer
        is only as defined as that order: sort first when it matters. Composed as
        :meth:`array_agg` followed by the list's own ``arg_max``. The *value* at another
        column's maximum is :meth:`max_by`.

        Returns:
            An Int64 expression over an aggregate, for use in ``agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [None, 5, 2, 5]})
                >>> ds.agg(r=bt.col("x").arg_max()).to_pydict()
                {'r': [1]}
        """
        from batcher.plan.expr_ir.func_nodes import ListFunc

        return ListFunc("arg_max", self.array_agg())

    def bool_and(self, *, empty_value: bool | None = None) -> AggExpr | Expr:
        """Logical AND of this boolean expression's non-null values per group.

        Null when the group has no non-null value; ``empty_value=True`` answers ``True``
        there, as Polars' ``all`` does.

        Args:
            empty_value: The result for a group with no non-null value; ``None`` keeps null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, True]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_and()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [False, True]}
        """
        agg = AggExpr("bool_and", self)
        if empty_value is None:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.with_empty_value(agg, empty_value)

    def bool_or(self, *, empty_value: bool | None = None) -> AggExpr | Expr:
        """Logical OR of this boolean expression's non-null values per group.

        Null when the group has no non-null value; ``empty_value=False`` answers ``False``
        there, as Polars' ``any`` does.

        Args:
            empty_value: The result for a group with no non-null value; ``None`` keeps null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [True, False, False]})
                >>> ds.group_by("g").agg(r=bt.col("x").bool_or()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [True, False]}
        """
        agg = AggExpr("bool_or", self)
        if empty_value is None:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.with_empty_value(agg, empty_value)

    def product(self, *, empty_value: float | None = None) -> AggExpr | Expr:
        """Product of non-null values per group (DuckDB ``product``; → Float64).

        Mergeable, so identical single-node and distributed. A group with no non-null
        value is null; ``empty_value=1`` answers ``1`` there, as Polars does.

        Args:
            empty_value: The result for a group with no non-null value; ``None`` keeps null.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 5]})
                >>> ds.group_by("g").agg(r=bt.col("x").product()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [6.0, 5.0]}
        """
        agg = AggExpr("product", self)
        if empty_value is None:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.with_empty_value(agg, float(empty_value))

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

    def array_agg(self, *, ignore_nulls: bool = False) -> AggExpr | Expr:
        """Collect each group's values (including nulls) into a ``List`` (SQL ``array_agg``).

        Like DuckDB ``array_agg``/``list``: null elements are kept, so a group of
        ``[10, None, 30]`` collects to ``[10, None, 30]``. An aggregate over zero rows
        (a global ``array_agg`` on an empty relation) is NULL, not ``[]``. Without an
        explicit order the element order is arrival-dependent. Mergeable — the per-group
        value list is the partial state, so the result is the same single-node and
        distributed.

        Chain a list reduction on the result column to summarize it, e.g.
        ``ds.group_by("g").agg(tags=col("t").array_agg())`` then
        ``col("tags").list.join(",")``.

        ``ignore_nulls=True`` leaves the nulls out, as Spark's ``collect_list`` and
        ``array_agg`` do, so a group of only nulls collects to ``[]``.

        Args:
            ignore_nulls: Whether to leave null values out of the list.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)`` or ``.over(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").array_agg()).sort("g").to_pydict()
                {'g': ['a', 'b'], 'r': [[1, 2], [10]]}

                >>> nulls = bt.from_pydict({"x": [1, None]})
                >>> nulls.agg(r=bt.col("x").array_agg(ignore_nulls=True)).to_pydict()
                {'r': [[1]]}
        """
        agg = AggExpr("list_agg", self)
        if not ignore_nulls:
            return agg
        from batcher.plan.functions import aggregate_semantics as sem

        return sem.array_agg_without_nulls(agg)

    # --- Cumulative / shift (Polars-style window conveniences) ------------------
    # Each returns a window expression (running aggregate / lag-lead), so use it in
    # `with_columns`/`select`. `partition_by` gives a per-group running value. An order is
    # required -- `order_by=` here or `.over(order_by=...)` -- because Batcher keeps no
    # arrival order across a parallel scan; `Window` refuses a running value without one.
    def _running(
        self,
        agg: str,
        partition_by: Iterable[IntoExpr],
        order_by: Iterable[IntoExpr],
        *,
        reverse: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """`agg` from the first row to this one, or from this one to the last (`reverse`).

        `propagate_nulls` answers null on a null input row while the running value carries
        on past it — Polars' reading — as a CASE over the same window, so it adds no IR.
        The null is ``nullif(running, running)``, a null of the running value's own type.
        """
        from batcher.plan.expr_ir.constructors import nullif, when

        frame = (0, None) if reverse else (None, 0)
        running = AggExpr(agg, self).over(partition_by=partition_by, order_by=order_by, frame=frame)
        if not propagate_nulls:
            return running
        return when(self.is_null()).then(nullif(running, running)).otherwise(running)

    def cum_sum(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        reverse: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """Cumulative (running) sum from the first row to the current one — Polars ``cum_sum``.

        Nulls are **skipped, not propagated**: a null leaves the running value
        unchanged, as SQL's window aggregate does and as :meth:`cum_prod` documents.
        Polars propagates instead, returning null at the null row and for it alone;
        ``propagate_nulls=True`` gives that reading.

        A window expression (one value per row, no row collapse) — use it in
        ``with_columns``/``select``. `order_by` is required, here or through
        ``.over(order_by=...)``: without it there is no defined running order.

        Args:
            partition_by: Restart the running sum per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            reverse: Accumulate from the last row back to the current one (Polars
                ``reverse=True``).
            propagate_nulls: Answer null on a null input row (Polars), while the running
                value still skips it for the rows after.

        Returns:
            A window expression carrying the running sum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1, 2, 3, 4]})
                >>> ds.with_columns(cs=bt.col("x").cum_sum(order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, 2, 3, 4], 'cs': [1, 3, 6, 10]}
        """
        return self._running(
            "sum", partition_by, order_by, reverse=reverse, propagate_nulls=propagate_nulls
        )

    def cum_min(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        reverse: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """Cumulative (running) minimum up to the current row — Polars ``cum_min``.

        Nulls are **skipped, not propagated**: a null leaves the running value
        unchanged, as SQL's window aggregate does and as :meth:`cum_prod` documents.
        Polars propagates instead, returning null at the null row and for it alone;
        ``propagate_nulls=True`` gives that reading.

        A window expression; use it in ``with_columns``/``select``. Pass
        `partition_by` to restart per group and `order_by` to set the running order.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            reverse: Accumulate from the last row back to the current one (Polars
                ``reverse=True``).
            propagate_nulls: Answer null on a null input row (Polars), while the running
                value still skips it for the rows after.

        Returns:
            A window expression carrying the running minimum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3, 4], "x": [3, 1, 4, 1, 5]})
                >>> ds.with_columns(cm=bt.col("x").cum_min(order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3, 4], 'x': [3, 1, 4, 1, 5], 'cm': [3, 1, 1, 1, 1]}
        """
        return self._running(
            "min", partition_by, order_by, reverse=reverse, propagate_nulls=propagate_nulls
        )

    def cum_max(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        reverse: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """Cumulative (running) maximum up to the current row — Polars ``cum_max``.

        Nulls are **skipped, not propagated**: a null leaves the running value
        unchanged, as SQL's window aggregate does and as :meth:`cum_prod` documents.
        Polars propagates instead, returning null at the null row and for it alone;
        ``propagate_nulls=True`` gives that reading.

        A window expression; use it in ``with_columns``/``select``. Pass
        `partition_by` to restart per group and `order_by` to set the running order.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            reverse: Accumulate from the last row back to the current one (Polars
                ``reverse=True``).
            propagate_nulls: Answer null on a null input row (Polars), while the running
                value still skips it for the rows after.

        Returns:
            A window expression carrying the running maximum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3, 4], "x": [3, 1, 4, 1, 5]})
                >>> ds.with_columns(cm=bt.col("x").cum_max(order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3, 4], 'x': [3, 1, 4, 1, 5], 'cm': [3, 3, 4, 4, 5]}
        """
        return self._running(
            "max", partition_by, order_by, reverse=reverse, propagate_nulls=propagate_nulls
        )

    def cum_prod(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        reverse: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """Cumulative (running) product up to the current row — Polars ``cum_prod``.

        Completes the running family beside :meth:`cum_sum`, :meth:`cum_min`,
        :meth:`cum_max` and :meth:`cum_count`. Nulls are skipped rather than
        propagated, so a null leaves the running product unchanged, which is what the
        other members of the family do and what SQL's ``product`` aggregate does.

        The result is ``Float64`` even for an integer input, because a running product
        overflows an ``Int64`` far sooner than a running sum and silently wrapping is
        the wrong answer to give a compounding factor.

        Args:
            partition_by: Restart the running value per group of these key expressions.
            order_by: Order rows by these expressions before accumulating.
            reverse: Accumulate from the last row back to the current one (Polars
                ``reverse=True``).
            propagate_nulls: Answer null on a null input row (Polars), while the running
                value still skips it for the rows after.

        Returns:
            A window expression carrying the running product.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2], "x": [2.0, 3.0, 4.0]})
                >>> ds.with_columns(cp=bt.col("x").cum_prod(order_by="t")).to_pydict()
                {'t': [0, 1, 2], 'x': [2.0, 3.0, 4.0], 'cp': [2.0, 6.0, 24.0]}

                >>> rates = bt.from_pydict(
                ...     {"fund": ["a", "a", "b", "b"], "r": [1.1, 1.2, 2.0, 0.5]}
                ... )
                >>> rates.with_columns(
                ...     growth=bt.col("r").cum_prod(partition_by="fund", order_by="r")
                ... ).to_pydict()["growth"]
                [1.1, 1.32, 1.0, 0.5]
        """
        return self._running(
            "product", partition_by, order_by, reverse=reverse, propagate_nulls=propagate_nulls
        )

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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [3, 1, 4, 1]})
                >>> ds.with_columns(cc=bt.col("x").cum_count(order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [3, 1, 4, 1], 'cc': [1, 2, 3, 4]}
        """
        return self._running("count", partition_by, order_by)

    def over(
        self,
        partition_by: Iterable[IntoExpr] | IntoExpr | None = (),
        order_by: Iterable[IntoExpr] | IntoExpr | None = (),
        frame: FrameSpec | None = None,
        *,
        descending: bool | Iterable[bool] = False,
        nulls_last: bool = True,
        mapping_strategy: str = "group_to_rows",
    ) -> Expr:
        """Evaluate this expression per window — Polars ``over``, SQL ``… OVER (…)``.

        Every aggregate inside the expression becomes the aggregate over its partition, and
        every window function inside it (``shift``, ``cum_sum``, ``rank``, ...) is bound to
        the partition and order. So ``(col("x") / col("x").sum()).over("g")`` is each
        row's share of its group, and ``col("x").shift().over("g", order_by="t")`` is the
        previous value within the group. An expression with no aggregate or window inside
        is returned unchanged, because a per-row value is the same computed per group.

        `order_by` is what order-dependent expressions (``shift``, ``diff``, ``cum_*``,
        ``first``/``last``, fills, EWMs) need and require. An inner window keeps its own
        partition and adds this one; this `order_by`, when given, replaces its own order.

        Unlike Polars, an `order_by` here also makes a plain aggregate *running*, as SQL's
        ``sum(x) OVER (ORDER BY t)`` is -- Polars ignores the order for an aggregate. Leave
        `order_by` off an expression whose aggregates should cover the whole partition.

        Args:
            partition_by: Key expressions or column names the window is computed within;
                empty for the whole frame.
            order_by: Key expressions or column names giving the row order.
            frame: ``(start, end)`` signed row offsets for the aggregates inside, as on
                :meth:`AggExpr.over`.
            descending: Order every `order_by` key, or each one, largest first.
            nulls_last: Sort null keys after the non-null ones (the SQL default; Polars
                defaults to nulls first).
            mapping_strategy: How a group's result maps back to rows. Only
                ``"group_to_rows"`` is supported; ``"join"`` and ``"explode"`` raise.

        Returns:
            The expression evaluated over the window.

        Raises:
            PlanError: For an unsupported `mapping_strategy`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "t": [1, 2, 1], "x": [1, 3, 10]})
                >>> share = (bt.col("x") / bt.col("x").sum()).over("g")
                >>> ds.with_columns(share=share).sort("g", "t").to_pydict()["share"]
                [0.25, 0.75, 1.0]
                >>> prev = bt.col("x").shift(1).over("g", order_by="t")
                >>> ds.with_columns(prev=prev).sort("g", "t").to_pydict()["prev"]
                [None, 1, None]
        """
        from batcher.plan.expr_rewrite.over import bind_over

        return bind_over(
            self,
            partition_by,
            order_by,
            frame,
            descending=descending,
            nulls_last=nulls_last,
            mapping_strategy=mapping_strategy,
        )

    def shift(self, n: int = 1) -> WindowExpr:
        """Shift values by `n` rows in row order — Polars ``shift`` (lag/lead).

        Positive `n` lags (moves down, vacated leading rows null); negative `n` leads
        (moves up). A window expression — use in ``with_columns``/``select``, bound to
        an order with ``.over(order_by=...)``, which is required.

        Args:
            n: Number of rows to shift; positive lags, negative leads.

        Returns:
            A window expression with the values shifted by `n` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1, 2, 3, 4]})
                >>> ds.with_columns(s=bt.col("x").shift(1).over(order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, 2, 3, 4], 's': [None, 1, 2, 3]}
        """
        from batcher.plan.expr_ir.nodes import lag, lead

        return lag(self, n) if n >= 0 else lead(self, -n)

    def _peak(
        self,
        greater: bool,
        partition_by: Iterable[IntoExpr],
        order_by: Iterable[IntoExpr],
        edges: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """A local extremum: strictly beyond both neighbours along the given order.

        Composed from `lag`/`lead` windows, so it adds no IR. The comparison is *strict*,
        which is what makes a plateau not a peak. By default it is also null-safe: a
        neighbouring null, or a missing neighbour at an edge, makes that side false.

        `edges` makes a missing neighbour true instead, so an edge row that beats its one
        neighbour is a peak. `propagate_nulls` makes a comparison against a null (or of a
        null row) unknown and combines the two sides with SQL's three-valued AND. An edge
        and a null neighbour both read as null from `lag`, so the edge is told apart by
        lagging ``is_null()``, which is itself never null."""
        from batcher.plan.expr_ir.constructors import lit, when
        from batcher.plan.expr_ir.nodes import lag, lead

        def side(window) -> Expr:
            neighbour = window(self, 1).over(partition_by=partition_by, order_by=order_by)
            at_edge = window(self.is_null(), 1).over(partition_by=partition_by, order_by=order_by)
            beats = self > neighbour if greater else self < neighbour
            if not propagate_nulls:
                beats = beats.fill_null(False)
            return when(at_edge.is_null()).then(Lit(edges)).otherwise(beats)

        peak = side(lag) & side(lead)
        if propagate_nulls:
            return when(self.is_null()).then(lit(None, "bool")).otherwise(peak)
        return peak

    def peak_max(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        edges: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """True at a local maximum — strictly above both neighbours (Polars ``peak_max``).

        The turning points of a series: the highs of a price trace, the spikes in a sensor
        reading, the local optima of a scan. The comparison is strict, so a plateau has no
        peak.

        **The first and last rows of a partition are never peaks**, because a peak is
        defined by the rows on *both* sides and an edge row has only one. Polars decides the
        edges differently — it counts a row that beats its single neighbour — so the two
        agree on every interior row and can differ on the two ends. Batcher takes the
        symmetric rule deliberately: it is the one that means the same thing for `peak_max`
        and `peak_min`, and the one that does not change an answer when a partition is
        split differently.

        Composed from two `shift` windows, so it adds no engine surface. `order_by` decides
        which rows are "neighbours" and is what makes the answer well defined.

        Polars' ``peak_max`` is ``edges=True, propagate_nulls=True``: an edge row that beats
        its one neighbour is a peak, and a comparison touching a null is null.

        Args:
            partition_by: Restart the neighbour comparison per group of these keys.
            order_by: Order rows by these expressions before comparing neighbours.
            edges: Whether a first or last row counts as beating its missing neighbour.
            propagate_nulls: Whether a null row or neighbour makes the answer null (SQL
                three-valued logic) rather than false.

        Returns:
            A Boolean expression, true at a local maximum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4, 5], "x": [1, 5, 2, 8, 3]})
                >>> ds.with_columns(p=bt.col("x").peak_max(order_by=["t"])).to_pydict()["p"]
                [False, True, False, True, False]
                >>> edges = bt.col("x").peak_max(order_by=["t"], edges=True)
                >>> ds.with_columns(p=edges).to_pydict()["p"]
                [False, True, False, True, True]
        """
        return self._peak(True, partition_by, order_by, edges, propagate_nulls)

    def peak_min(
        self,
        *,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
        edges: bool = False,
        propagate_nulls: bool = False,
    ) -> Expr:
        """True at a local minimum — strictly below both neighbours (Polars ``peak_min``).

        The mirror of :meth:`peak_max`; see it for the strictness and the edge convention.
        Polars' ``peak_min`` is ``propagate_nulls=True`` with the default ``edges=False``:
        unlike its ``peak_max``, it never counts an edge row.

        Args:
            partition_by: Restart the neighbour comparison per group of these keys.
            order_by: Order rows by these expressions before comparing neighbours.
            edges: Whether a first or last row counts as beating its missing neighbour.
            propagate_nulls: Whether a null row or neighbour makes the answer null (SQL
                three-valued logic) rather than false.

        Returns:
            A Boolean expression, true at a local minimum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4, 5], "x": [5, 1, 4, 0, 6]})
                >>> ds.with_columns(p=bt.col("x").peak_min(order_by=["t"])).to_pydict()["p"]
                [False, True, False, True, False]
        """
        return self._peak(False, partition_by, order_by, edges, propagate_nulls)

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

    def interpolate(self) -> WindowExpr:
        """Draw a straight line across each interior gap — Polars ``interpolate``.

        Where :meth:`forward_fill` holds the last reading flat across a gap, this
        assumes the quantity moved steadily and reconstructs the path: a null bracketed
        by non-null values at ordered positions ``a`` and ``b`` takes the point on the
        segment between them, weighted by how far along the gap it sits. Use it for a
        continuous physical signal (a temperature, a level, a cumulative counter);
        prefer a fill for a state that genuinely holds between reports.

        Leading and trailing nulls have nothing to interpolate *between* and stay null.
        The result is always floating point, because the value between two integers
        generally is not one.

        A window expression, so it must be bound with ``.over(...)`` and ``order_by``
        is required — interpolation follows a defined row order, and an unordered
        relation has none.

        Returns:
            A window expression carrying the interpolated column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4], "x": [10.0, None, None, 40.0]})
                >>> ds.with_columns(i=bt.col("x").interpolate().over(order_by=["t"])).to_pydict()
                {'t': [1, 2, 3, 4], 'x': [10.0, None, None, 40.0], 'i': [10.0, 20.0, 30.0, 40.0]}
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        return WindowExpr("interpolate", self, [], [], None)

    def rle_id(self) -> WindowExpr:
        """Number the runs of equal consecutive values — Polars ``rle_id``.

        Each row gets the 0-based index of the run it belongs to, incrementing every
        time the value differs from the previous row's along the order. It is the
        segmentation primitive behind "how long has this machine been in its current
        state" and "split this series wherever the regime changed": group by the run id
        to collapse each run to a row, or count within it to measure a run's length.

        Consecutive nulls form one run, and a value that returns after a gap opens a
        *new* run rather than rejoining the earlier one.

        A window expression, so it must be bound with ``.over(...)`` and ``order_by``
        is required.

        Returns:
            A window expression carrying the 0-based run index.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4], "s": ["on", "on", "off", "on"]})
                >>> ds.with_columns(r=bt.col("s").rle_id().over(order_by=["t"])).to_pydict()
                {'t': [1, 2, 3, 4], 's': ['on', 'on', 'off', 'on'], 'r': [0, 0, 1, 2]}
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        return WindowExpr("rle_id", self, [], [], None)

    # --- exponentially weighted moving statistics ---------------------------
    def _ewm(
        self,
        func: str,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
    ) -> WindowExpr:
        """Resolve one of the four decay spellings to an alpha and build the window."""
        from batcher.plan.expr_ir.nodes import WindowExpr

        resolved = _ewm_alpha(func, com, span, half_life, alpha)
        return WindowExpr(func, self, [], [], None, alpha=resolved)

    def ewm_mean(
        self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
    ) -> WindowExpr:
        """Exponentially weighted moving average — Polars/pandas ``ewm_mean``.

        Where :meth:`rolling_mean` gives every row in a fixed window the same weight and
        forgets everything older, an EWM weights row ``i`` by ``(1-alpha)^(t-i)``: recent
        readings dominate, old ones fade smoothly rather than dropping off a cliff. That
        makes it the standard smoother for a noisy sensor or price series, and the basis
        of MACD and similar indicators.

        Give the decay exactly one way. All four are the same number spelled for
        different habits: ``alpha`` directly, ``span`` (``alpha = 2/(span+1)``, the
        "N-period EMA" of technical analysis), ``half_life`` (the lag at which a
        reading's weight halves), or ``com`` (centre of mass,
        ``alpha = 1/(1+com)``).

        A null input row yields a null output and contributes no value, but still ages
        the decay — pandas' ``adjust=True, ignore_na=False`` and Polars'
        ``adjust=True, ignore_nulls=False``, the default in both.

        A window expression, so it must be bound with ``.over(...)`` and ``order_by``
        is required.

        Args:
            com: Centre of mass, ``>= 0``.
            span: Span, ``>= 1``.
            half_life: Half-life in rows, ``> 0``.
            alpha: The smoothing factor itself, in ``(0, 1]``.

        Returns:
            A window expression carrying the exponentially weighted mean.

        Raises:
            PlanError: If none or more than one of the four is given, or one is out of
                range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
                >>> w = bt.col("x").ewm_mean(alpha=0.5).over(order_by=["t"])
                >>> ds.with_columns(e=w).to_pydict()["e"]
                [1.0, 1.6666666666666665, 2.4285714285714284]
        """
        return self._ewm("ewm_mean", com, span, half_life, alpha)

    def ewm_mean_by(
        self,
        by: IntoExpr,
        half_life: str | int | float,
        *,
        partition_by: Iterable[IntoExpr] = (),
    ) -> WindowExpr:
        """Exponentially weighted mean decayed by *elapsed time* — Polars ``ewm_mean_by``.

        :meth:`ewm_mean` decays once per row, which is right only when the readings are
        evenly spaced. An irregular feed breaks it: an hour of silence costs exactly the
        weight one second would, so a burst of samples dominates a quiet stretch that
        actually lasted longer. Here the weight is ``exp(-ln2 · Δt / half_life)``, where
        ``Δt`` is the real gap in the `by` column — so the smoother says the same thing
        whatever the sampling rate did.

        `half_life` is the lag at which a reading's weight halves: give a duration
        (``"5m"``) for a timestamp or date column, and a number in the column's own units
        for a numeric one.

        A null value yields a null output and leaves the anchor where it was, so the next
        reading decays from the last one actually seen rather than from an empty row.

        A window expression, so it must be bound with ``.over(...)``. `by` becomes its
        order, which is what makes the gap well defined.

        Args:
            by: The single numeric or temporal column the decay is measured along.
            half_life: The lag at which a weight halves, as a duration or a number.
            partition_by: Restart the recurrence per group of these key expressions.

        Returns:
            A window expression carrying the time-decayed exponentially weighted mean.

        Raises:
            PlanError: If `half_life` is not a positive fixed-length duration or number.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> base = dt.datetime(2024, 1, 1)
                >>> ds = bt.from_pydict(
                ...     {
                ...         "t": [base, base + dt.timedelta(minutes=1),
                ...               base + dt.timedelta(minutes=5)],
                ...         "x": [1.0, 2.0, 3.0],
                ...     }
                ... )
                >>> ds.with_columns(e=bt.col("x").ewm_mean_by("t", "2m")).to_pydict()["e"]
                [1.0, 1.2928932188134525, 2.573223304703363]
        """
        from batcher.plan.expr_ir.nodes import WindowExpr
        from batcher.plan.functions.temporal import _duration_micros

        if isinstance(half_life, str):
            hl = float(_duration_micros(half_life, arg="ewm_mean_by half_life"))
        else:
            hl = require_float(half_life, func="ewm_mean_by", arg="half_life")
            if hl <= 0:
                raise PlanError(f"ewm_mean_by(): half_life must be > 0, got {half_life!r}")
        return WindowExpr("ewm_mean", self, [], [], None, half_life=hl).over(
            partition_by=partition_by, order_by=[by]
        )

    def ewm_std(
        self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
    ) -> WindowExpr:
        """Exponentially weighted moving standard deviation — Polars ``ewm_std``.

        The spread counterpart of :meth:`ewm_mean`, over the same decaying weights: a
        recent burst of noise widens it quickly and a quiet stretch narrows it, which is
        what makes it usable as a live volatility or control-limit estimate.

        This is the *sample* form, debiased by the weights, so the first row of a
        partition is null — a single observation has no spread, exactly as
        ``col("x").std()`` over a one-row window is null. (Polars reports ``0.0`` there;
        pandas reports null, and this follows pandas and Batcher's own ``var``/``std``.)

        Args:
            com: Centre of mass, ``>= 0``.
            span: Span, ``>= 1``.
            half_life: Half-life in rows, ``> 0``.
            alpha: The smoothing factor itself, in ``(0, 1]``.

        Returns:
            A window expression carrying the exponentially weighted standard deviation.

        Raises:
            PlanError: If none or more than one of the four is given, or one is out of
                range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
                >>> w = bt.col("x").ewm_std(alpha=0.5).over(order_by=["t"])
                >>> ds.with_columns(e=w).to_pydict()["e"]
                [None, 0.7071067811865477, 0.9636241116594317]
        """
        return self._ewm("ewm_std", com, span, half_life, alpha)

    def ewm_var(
        self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
    ) -> WindowExpr:
        """Exponentially weighted moving variance — Polars ``ewm_var``.

        The square of :meth:`ewm_std`, sharing its decay spellings and its null first
        row. Prefer it when the value feeds further arithmetic (a z-score, a Kalman-style
        update) and the square root would only be undone.

        Args:
            com: Centre of mass, ``>= 0``.
            span: Span, ``>= 1``.
            half_life: Half-life in rows, ``> 0``.
            alpha: The smoothing factor itself, in ``(0, 1]``.

        Returns:
            A window expression carrying the exponentially weighted variance.

        Raises:
            PlanError: If none or more than one of the four is given, or one is out of
                range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
                >>> w = bt.col("x").ewm_var(alpha=0.5).over(order_by=["t"])
                >>> ds.with_columns(e=w).to_pydict()["e"]
                [None, 0.5000000000000002, 0.928571428571429]
        """
        return self._ewm("ewm_var", com, span, half_life, alpha)

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

        window_size = require_int(window_size, func=f"rolling_{agg}", arg="window_size", minimum=1)
        if min_periods is not None:
            min_periods = require_int(min_periods, func=f"rolling_{agg}", arg="min_periods")
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1, 2, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_sum(2, order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, 2, 3, 4], 'r': [1, 3, 5, 7]}
                >>> r = bt.col("x").rolling_sum(2, min_periods=2, order_by="t")
                >>> ds.with_columns(r=r).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, 2, 3, 4], 'r': [None, 3, 5, 7]}
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1, 2, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_mean(2, order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, 2, 3, 4], 'r': [1.0, 1.5, 2.5, 3.5]}
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [3, 1, 4, 1]})
                >>> ds.with_columns(r=bt.col("x").rolling_min(2, order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [3, 1, 4, 1], 'r': [3, 1, 1, 1]}
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [3, 1, 4, 1]})
                >>> ds.with_columns(r=bt.col("x").rolling_max(2, order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [3, 1, 4, 1], 'r': [3, 3, 4, 4]}
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1, None, 3, 4]})
                >>> ds.with_columns(r=bt.col("x").rolling_count(2, order_by="t")).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1, None, 3, 4], 'r': [1, 1, 1, 2]}
        """
        return self._rolling("count", window_size, min_periods, partition_by, order_by)

    # --- rolling over a time window, not a row count -----------------------
    def _rolling_by(
        self,
        agg: str,
        by: IntoExpr,
        window_size: str | int,
        min_periods: int | None,
        partition_by: Iterable[IntoExpr],
    ) -> Expr:
        """`agg` over the rows within `window_size` of the current row's `by` value.

        A `RANGE` frame of ``(-width, 0)`` ordered by `by`, where `width` is the window
        in the key's own units — microseconds for a temporal key, so a duration string
        resolves through the same parser `bt.window` uses and the two cannot disagree
        about how long a minute is. Everything else (partial leading frames,
        `min_periods`) is `_rolling`'s, over a different frame."""
        from batcher.plan.expr_ir.constructors import nullif, when
        from batcher.plan.functions.temporal import _duration_micros

        if isinstance(window_size, str):
            width = _duration_micros(window_size, arg=f"rolling_{agg}_by window_size")
        else:
            width = require_int(window_size, func=f"rolling_{agg}_by", arg="window_size", minimum=1)
        if min_periods is not None:
            min_periods = require_int(min_periods, func=f"rolling_{agg}_by", arg="min_periods")
            if min_periods < 1:
                raise PlanError(f"rolling_{agg}_by(): min_periods must be >= 1, got {min_periods}")
        frame = (-width, 0, "range")
        value = AggExpr(agg, self).over(partition_by=partition_by, order_by=[by], frame=frame)
        if min_periods is None:
            return value
        seen = AggExpr("count", self).over(partition_by=partition_by, order_by=[by], frame=frame)
        return when(seen >= Lit(min_periods)).then(value).otherwise(nullif(value, value))

    def rolling_sum_by(
        self,
        by: IntoExpr,
        window_size: str | int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Sum over a *time* window ending at this row — Polars ``rolling_sum_by``.

        Where :meth:`rolling_sum` counts rows, this counts along the `by` column's values:
        ``rolling_sum_by("ts", "5m")`` is the last five minutes however many readings that
        turns out to be. That is the difference between a moving average that means the
        same thing at every sampling rate and one that silently widens whenever a sensor
        goes quiet.

        It is SQL's ``RANGE BETWEEN <window_size> PRECEDING AND CURRENT ROW``, so **both
        endpoints are included** — Polars' ``closed="both"``, not its ``closed="right"``
        default. Rows exactly `window_size` back are in the window.

        `by` must be a single numeric or temporal column, because the bound is arithmetic
        on it. Give `window_size` as a duration (``"5m"``, ``"1h30m"``) for a timestamp or
        date column, and as a number for a numeric one.

        Args:
            by: The single column whose values the window is measured in.
            window_size: The window width, as a duration string or a number.
            min_periods: Least non-null values the window must hold, else the row is null.
            partition_by: Restart the window per group of these key expressions.

        Returns:
            The rolling sum over the time window.

        Raises:
            PlanError: If `window_size` is not a positive fixed-length duration or count,
                or `min_periods` is below 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> base = dt.datetime(2024, 1, 1, 0, 0)
                >>> ds = bt.from_pydict(
                ...     {
                ...         "ts": [base, base + dt.timedelta(minutes=1),
                ...                base + dt.timedelta(minutes=30)],
                ...         "v": [1, 2, 4],
                ...     }
                ... )
                >>> ds.with_columns(r=bt.col("v").rolling_sum_by("ts", "5m")).to_pydict()["r"]
                [1, 3, 4]
        """
        return self._rolling_by("sum", by, window_size, min_periods, partition_by)

    def rolling_mean_by(
        self,
        by: IntoExpr,
        window_size: str | int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Mean over a time window ending at this row — Polars ``rolling_mean_by``.

        See :meth:`rolling_sum_by` for how the window is measured and for the inclusive
        endpoint rule.

        Args:
            by: The single column whose values the window is measured in.
            window_size: The window width, as a duration string or a number.
            min_periods: Least non-null values the window must hold, else the row is null.
            partition_by: Restart the window per group of these key expressions.

        Returns:
            The rolling mean over the time window.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 60, 1800], "v": [1.0, 3.0, 5.0]})
                >>> ds.with_columns(r=bt.col("v").rolling_mean_by("t", 300)).to_pydict()["r"]
                [1.0, 2.0, 5.0]
        """
        return self._rolling_by("mean", by, window_size, min_periods, partition_by)

    def rolling_min_by(
        self,
        by: IntoExpr,
        window_size: str | int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Minimum over a time window ending at this row — Polars ``rolling_min_by``.

        See :meth:`rolling_sum_by` for how the window is measured.

        Args:
            by: The single column whose values the window is measured in.
            window_size: The window width, as a duration string or a number.
            min_periods: Least non-null values the window must hold, else the row is null.
            partition_by: Restart the window per group of these key expressions.

        Returns:
            The rolling minimum over the time window.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 60, 1800], "v": [5, 3, 9]})
                >>> ds.with_columns(r=bt.col("v").rolling_min_by("t", 300)).to_pydict()["r"]
                [5, 3, 9]
        """
        return self._rolling_by("min", by, window_size, min_periods, partition_by)

    def rolling_max_by(
        self,
        by: IntoExpr,
        window_size: str | int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Maximum over a time window ending at this row — Polars ``rolling_max_by``.

        See :meth:`rolling_sum_by` for how the window is measured.

        Args:
            by: The single column whose values the window is measured in.
            window_size: The window width, as a duration string or a number.
            min_periods: Least non-null values the window must hold, else the row is null.
            partition_by: Restart the window per group of these key expressions.

        Returns:
            The rolling maximum over the time window.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 60, 1800], "v": [5, 3, 9]})
                >>> ds.with_columns(r=bt.col("v").rolling_max_by("t", 300)).to_pydict()["r"]
                [5, 5, 9]
        """
        return self._rolling_by("max", by, window_size, min_periods, partition_by)

    def rolling_count_by(
        self,
        by: IntoExpr,
        window_size: str | int,
        *,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Count of non-null values over a time window — Polars ``rolling_count_by``.

        The natural way to ask "how many events in the last five minutes", and the guard
        that tells you whether a rolling mean over the same window is worth trusting.
        See :meth:`rolling_sum_by` for how the window is measured.

        Args:
            by: The single column whose values the window is measured in.
            window_size: The window width, as a duration string or a number.
            min_periods: Least non-null values the window must hold, else the row is null.
            partition_by: Restart the window per group of these key expressions.

        Returns:
            The rolling count over the time window.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 60, 120, 1800], "v": [1, 2, 3, 4]})
                >>> ds.with_columns(r=bt.col("v").rolling_count_by("t", 300)).to_pydict()["r"]
                [1, 2, 3, 1]
        """
        return self._rolling_by("count", by, window_size, min_periods, partition_by)

    def _rolling_var(
        self,
        window_size: int,
        ddof: int,
        min_periods: int | None,
        partition_by: Iterable[IntoExpr],
        order_by: Iterable[IntoExpr],
    ) -> Expr:
        """Sample/population variance over the trailing frame, on the framed variance kernel.

        The frame is `_rolling`'s, so the leading partial frames and `min_periods` behave as
        :meth:`rolling_sum`'s do. The sample variance comes from the engine's windowed
        ``var`` (`bc_runtime::window::agg`), which slides Welford moments through a two-stack
        fold: nothing is subtracted, so it neither cancels on large offsets nor lets a NaN
        outlive the windows that hold it. Any other `ddof` rescales that exactly,
        ``var_samp * (n - 1) / (n - ddof)``, and a frame of no more than `ddof` values has no
        statistic, so it is null (DuckDB's ``var_samp`` of one value is NULL).

        This replaced a composition over the centred moments ``E[x^2] - E[x]^2``. Centring
        on the partition mean kept small offsets exact, but it could not keep a frame of
        ``[1e9, 1e9 + 2]`` from cancelling to 0 in a partition that also held small values,
        and one NaN anywhere made the centre, and so every row of the partition, NaN."""
        from batcher.plan.expr_ir.constructors import nullif, when

        samp = self._rolling("var", window_size, min_periods, partition_by, order_by)
        if ddof == 1:
            return samp
        count = self._rolling("count", window_size, min_periods, partition_by, order_by).cast(
            "float64"
        )
        # `var_samp` is null on one value, where the centred sum of squares is 0.
        spread = Coalesce([samp * (count - Lit(1.0)), Lit(0.0)])
        scaled = spread / (count - Lit(float(ddof)))
        return when(count > Lit(float(ddof))).then(scaled).otherwise(nullif(scaled, scaled))

    def rolling_var(
        self,
        window_size: int,
        *,
        ddof: int = 1,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Variance over the `window_size` rows ending at the current one — Polars ``rolling_var``.

        Composed from windowed moments over the trailing frame (see :meth:`rolling_sum`
        for framing). ``ddof=1`` (the default) is the sample variance; ``ddof=0`` is the
        population variance. A degenerate frame holding fewer than ``ddof + 1`` values is
        undefined and yields NaN — pass `min_periods` to make those rows null instead.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            ddof: Delta degrees of freedom; ``1`` for sample, ``0`` for population.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling variance.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0]})
                >>> v = bt.col("x").rolling_var(2, min_periods=2, order_by="t")
                >>> ds.with_columns(v=v).to_pydict()
                {'t': [0, 1, 2, 3], 'x': [1.0, 2.0, 3.0, 4.0], 'v': [None, 0.5, 0.5, 0.5]}
        """
        return self._rolling_var(window_size, ddof, min_periods, partition_by, order_by)

    def rolling_std(
        self,
        window_size: int,
        *,
        ddof: int = 1,
        min_periods: int | None = None,
        partition_by: Iterable[IntoExpr] = (),
        order_by: Iterable[IntoExpr] = (),
    ) -> Expr:
        """Standard deviation over the `window_size` rows ending at the current one.

        The square root of :meth:`rolling_var`; ``ddof`` and `min_periods` behave as they
        do there. Polars ``rolling_std``.

        Args:
            window_size: How many rows the trailing frame spans, including this one.
            ddof: Delta degrees of freedom; ``1`` for sample, ``0`` for population.
            min_periods: Least non-null values the frame must hold, else the row is null.
            partition_by: Restart the frame per group of these key expressions.
            order_by: Order rows by these expressions before framing.

        Returns:
            A window expression carrying the rolling standard deviation.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": [0, 1, 2], "x": [2.0, 4.0, 6.0]})
                >>> s = bt.col("x").rolling_std(2, min_periods=2, order_by="t")
                >>> ds.with_columns(s=s).to_pydict()["s"]
                [None, 1.4142135623730951, 1.4142135623730951]
        """
        return self._rolling_var(window_size, ddof, min_periods, partition_by, order_by).sqrt()

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
                >>> ds = bt.from_pydict({"t": [0, 1, 2], "x": [1, 3, 8]})
                >>> ds.with_columns(d=bt.col("x").diff(order_by="t")).to_pydict()
                {'t': [0, 1, 2], 'x': [1, 3, 8], 'd': [None, 2, 5]}
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
                >>> ds = bt.from_pydict({"t": [0, 1, 2], "x": [10, 15, 30]})
                >>> ds.with_columns(p=bt.col("x").pct_change(order_by="t")).to_pydict()
                {'t': [0, 1, 2], 'x': [10, 15, 30], 'p': [None, 0.5, 1.0]}
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
        propagate_nulls: bool = False,
    ) -> Expr:
        """Rank the rows by this expression's value — SQL ``RANK() OVER (ORDER BY self)``.

        A window expression; use it in ``with_columns``/``select``. Ranks start at 1, and
        null values sort last and are ranked like any other value unless
        `propagate_nulls` is set. Polars ``rank()`` is ``method="average",
        propagate_nulls=True``.

        Args:
            method: How ties are numbered. ``"min"`` gives tied rows the same rank and
                leaves a gap (SQL ``RANK``); ``"dense"`` gives the same rank with no gap
                (``DENSE_RANK``); ``"ordinal"`` breaks ties arbitrarily so every row gets
                a distinct rank (``ROW_NUMBER``); ``"max"`` gives tied rows the highest
                rank they span; ``"average"`` gives them the mean of that span, as
                Float64 (Polars' default).
            descending: Rank from the largest value down instead of the smallest up.
            partition_by: Rank within each group of these key expressions.
            propagate_nulls: A null value gets a null rank (Polars, pandas
                ``na_option="keep"``).

        Returns:
            The 1-based rank of each row.

        Raises:
            PlanError: If `method` is not one of ``average``/``dense``/``max``/``min``/
                ``ordinal``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [10, 30, 10]})
                >>> ds.with_columns(r=bt.col("x").rank()).to_pydict()
                {'x': [10, 30, 10], 'r': [1, 3, 1]}

                >>> ds = bt.from_pydict({"x": [3, 1, 3, None]})
                >>> ds.with_columns(
                ...     r=bt.col("x").rank("average", propagate_nulls=True)
                ... ).to_pydict()["r"]
                [2.5, 1.0, 2.5, None]
        """
        from batcher.plan.expr_ir.constructors import nullif, when
        from batcher.plan.expr_ir.nodes import dense_rank, rank, row_number

        fns = {"min": rank, "dense": dense_rank, "ordinal": row_number}
        if method not in (*fns, "average", "max"):
            raise PlanError(
                "rank(): method must be one of ['average', 'dense', 'max', 'min', 'ordinal'], "
                f"got {method!r}"
            )
        ranked: Expr = fns.get(method, rank)().over(
            partition_by=partition_by, order_by=[(self, descending)]
        )
        if method in ("average", "max"):
            # A tie's span is its peer count: the rows sharing this value in the partition.
            # Counting `self IS NULL` counts every row, a null value's peers included.
            peers = AggExpr("count", IsNull(self)).over(
                partition_by=[*normalize_key_list(partition_by), self]
            )
            if method == "max":
                ranked = ranked + peers - Lit(1)
            else:
                ranked = ranked.cast("float64") + (peers - Lit(1)).cast("float64") / Lit(2.0)
        if propagate_nulls:
            return when(self.is_null()).then(nullif(ranked, ranked)).otherwise(ranked)
        return ranked

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
        return AggExpr("count", Lit(1)).over(partition_by=[self], frame=(None, None))


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


def _ewm_alpha(
    func: str,
    com: float | None,
    span: float | None,
    half_life: float | None,
    alpha: float | None,
) -> float:
    """Resolve the four EWM decay spellings to the single smoothing factor.

    pandas and Polars both accept `com`/`span`/`half_life`/`alpha` and require exactly
    one, because they are one number in four idioms: a trader says "12-period EMA"
    (`span`), a physicist says "half-life", a statistician says "centre of mass". The
    conversion is done once, here, so the IR and the engine carry a single alpha rather
    than four fields and a precedence rule.
    """
    given = {"com": com, "span": span, "half_life": half_life, "alpha": alpha}
    named = [k for k, v in given.items() if v is not None]
    if len(named) != 1:
        raise PlanError(
            f"{func}(): give exactly one of com, span, half_life, alpha — "
            f"got {', '.join(named) if named else 'none'}"
        )
    (key,) = named
    value = require_float(given[key], func=func, arg=key)
    if key == "alpha":
        resolved = value
    elif key == "com":
        if value < 0.0:
            raise PlanError(f"{func}(): com must be >= 0, got {value}")
        resolved = 1.0 / (1.0 + value)
    elif key == "span":
        if value < 1.0:
            raise PlanError(f"{func}(): span must be >= 1, got {value}")
        resolved = 2.0 / (value + 1.0)
    else:
        if value <= 0.0:
            raise PlanError(f"{func}(): half_life must be > 0, got {value}")
        resolved = 1.0 - math.exp(-math.log(2.0) / value)
    if not 0.0 < resolved <= 1.0:
        raise PlanError(f"{func}(): {key}={value} gives alpha {resolved}, outside (0, 1]")
    return resolved


def _as_exact_float(value: object) -> object:
    """A `Decimal` as a float when the float is the same number; anything else unchanged.

    `Decimal` is the type money is stored in, so `price > Decimal("9.99")` is the natural way
    -- and the only exact way -- to write the most common predicate in analytics. It raised
    `unsupported literal type: Decimal`, from `to_ir`, so the traceback pointed at `collect()`
    rather than at the `filter`. The column type works fine; only the literal was missing.

    The IR has no decimal literal, and adding one is a two-sided change to the wire contract.
    What is available is that the engine already compares a decimal *column* against a float
    literal correctly: `0.10`, `2.675` and `999999999.99` each match DuckDB exactly, because
    both sides land on the same float. That holds right up until the decimal carries more
    significant digits than a float64 can distinguish, where it silently stops matching --
    a 25-digit decimal compares equal to nothing at all.

    Silently wrong on money is far worse than unsupported, so the conversion is made only when
    it is provably lossless: the float must round-trip back to the same number. Everything
    inside float64's ~15 significant digits converts, which is every currency amount, rate and
    price; anything wider raises a typed error naming the limit instead of quietly answering
    the wrong question.

    Args:
        value: The literal a caller passed.

    Returns:
        An equal `float` for a round-tripping `Decimal`, else `value` unchanged.

    Raises:
        PlanError: If `value` is a `Decimal` that no float represents exactly.
    """
    if not isinstance(value, _decimal.Decimal):
        return value
    if not value.is_finite():
        return float(value)
    as_float = float(value)
    if _decimal.Decimal(repr(as_float)) == value:
        return as_float
    raise PlanError(
        f"decimal literal {value} has more significant digits than a 64-bit float can "
        f"represent, and the plan IR has no exact decimal literal. Compare against a float "
        f"or a string instead, or round the literal to 15 significant digits."
    )


def _as_python_scalar(value: object) -> object:
    """An array scalar as its plain Python equivalent; anything else unchanged.

    NumPy scalars are what real data work produces — `arr.max()`, `np.percentile(...)`,
    a pandas `Series.max()`, the result of any integer arithmetic on an array — and they
    reach the API constantly as filter thresholds. Almost none of them are subclasses of
    the Python types the wire encoder dispatches on: `numpy.float64` subclasses `float` and
    so worked by accident, while `numpy.int64`, every other width, `numpy.bool_` and a
    0-d array did not and raised `unsupported literal type: int64` — from `to_ir`, which
    on a lazy API means the traceback points at `collect()` rather than at the `filter`
    that built it.

    `.item()` is the conversion the array protocol already defines, and it lands each one on
    exactly the type the encoder below wants — including `numpy.datetime64`, which becomes a
    `datetime`/`date` the ladder already handles.

    Recognised structurally (`.item()` plus a `dtype`) rather than by importing NumPy: `plan`
    is the neutral contract layer and must not take a hard dependency on it. A 0-d array
    converts; an array with dimensions is left alone, so it still fails as the non-scalar it
    is instead of being silently reduced to its first element.

    Args:
        value: The literal a caller passed.

    Returns:
        The Python equivalent for an array scalar, else `value` unchanged.
    """
    item = getattr(value, "item", None)
    if not callable(item) or not hasattr(value, "dtype"):
        return value
    if getattr(value, "ndim", 0) != 0:
        return value  # a real array is not a scalar; let it fail as one
    try:
        return item()
    except (ValueError, TypeError):  # pragma: no cover - an exotic dtype with no Python form
        return value


class Lit(Expr):
    """A constant literal. The wire kind is inferred from the Python type."""

    # `_ir_cache` mirrors the memo `IRNode` keeps in its instance `__dict__`. `Lit` is
    # `__slots__`-based (there are more literals in a plan than any other node kind, and
    # a per-instance dict on each is real memory), so it needs the slot declared to get
    # the same one-lowering-per-node behavior every other node already has.
    __slots__ = ("_ir_cache", "value")

    def __init__(self, value: int | float | bool | str) -> None:
        """Wrap a Python scalar (or date/datetime) as a literal expression node."""
        self.value = _as_exact_float(_as_python_scalar(value))
        self._ir_cache: dict[str, Any] | None = None

    def to_ir(self) -> dict[str, Any]:
        """Lower this literal to its JSON IR dict (the Rust wire contract)."""
        cached = self._ir_cache
        if cached is not None:
            return cached
        v = self.value
        kind = type(v)
        # Exact-type dispatch for the four scalar kinds that make up almost every literal
        # in a plan, before the `isinstance` ladder the subclass relationships require
        # (bool before int, datetime before date). A subclass of one of these — an
        # `IntEnum` — still falls through to the ladder below and is tagged exactly as it
        # was. An **array scalar** is not a subclass and never reached the ladder at all;
        # `__init__` converts it before it gets here (see `_as_python_scalar`).
        if kind is int:
            tagged: dict[str, Any] = {"int": v}
        elif kind is str:
            tagged = {"str": v}
        elif kind is bool:
            tagged = {"bool": v}
        elif kind is float and -math.inf < v < math.inf:
            tagged = {"float": v}  # finite: the numeric wire form
        elif isinstance(v, bool):
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
            elif v == math.inf:
                tagged = {"float": "inf"}
            elif v == -math.inf:
                tagged = {"float": "-inf"}
            else:
                tagged = {"float": v}
        elif isinstance(v, str):
            tagged = {"str": v}
        elif isinstance(v, _dt.datetime):
            # Microseconds since the Unix epoch. A tz-naive datetime is the wall clock (matching
            # how pyarrow stores tz-naive Timestamp(us) columns); a tz-aware one is its UTC
            # instant. Subtract a *matching* epoch — a UTC-aware epoch for an aware datetime —
            # so an aware literal doesn't raise "can't subtract offset-naive and offset-aware
            # datetimes" (which crashed `col("ts") > lit(aware_datetime)`), and its micros land
            # on the true UTC instant that the engine's tz-aware comparison expects.
            epoch = (
                _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
                if v.tzinfo is not None
                else _dt.datetime(1970, 1, 1)
            )
            delta = v - epoch
            micros = delta.days * MICROS_PER_DAY + delta.seconds * 1_000_000 + delta.microseconds
            tagged = {"timestamp": micros}
        elif isinstance(v, _dt.date):
            tagged = {"date": (v - _dt.date(1970, 1, 1)).days}
        elif isinstance(v, _dt.time):
            # Lowered as a cast from its ISO text, which is exactly what the SQL parser
            # emits for ``TIME '01:02:03'``. Not a tagged literal like `date` and
            # `timestamp` above, because `bc_expr::Literal` has no `Time` variant — giving
            # it one would be a two-sided IR change across the FFI for something the engine
            # already evaluates correctly by this route.
            #
            # Until this existed, ``col("t") > time(1, 0)`` raised ``unsupported literal
            # type: time`` while ``WHERE t > TIME '01:00:00'`` answered it — the same query,
            # over the same engine, working through one front-end and not the other.
            if v.tzinfo is not None:
                raise TypeError(
                    "a time literal cannot carry a timezone: arrow's time64 has no zone, "
                    "so the offset would be silently dropped. Use a datetime for an "
                    "instant, or a naive time for a wall-clock time of day."
                )
            return Cast(Lit(v.isoformat()), "time").to_ir()
        else:
            # Names the value and the remedy, not just its type. This is reached
            # whenever a non-literal object is used where a constant is expected
            # (``col("x") == some_object``), and "unsupported literal type: Foo" left
            # the reader to work out both which argument and what to do instead.
            raise PlanError(
                f"cannot use {type(v).__name__} {v!r} as a literal value: a literal must "
                "be a string, number, boolean, None, or a date/time/datetime/Decimal. "
                "To reference a column use col('name'); to pass a Python object to your "
                "own code use map_batches()."
            )
        out = {"e": ExprTag.LIT, "value": tagged}
        self._ir_cache = out
        return out


def int_literal(expr: Expr) -> int | None:
    """The Python `int` a plain integer literal holds, or `None` if it is not one.

    The `bool` check is the whole point and the reason this is shared rather than rewritten
    per caller: `bool` subclasses `int` in Python, so `isinstance(Lit(True).value, int)` is
    true and a rule that skips the guard silently treats `WHERE flag = TRUE` as `= 1`. Five
    Kyber rule families each carried their own copy of this function -- `_int_lit` three
    times, plus `_int_literal` and `_seconds_literal` -- byte-identical including the guard.
    Five copies of one subtlety is five chances for four of them to be left behind by a fix.

    Lives beside `Lit` in the neutral `plan` layer rather than in a Kyber helpers module
    because every caller already imports `plan.expr_ir`, so sharing it adds no import edge --
    and an edge into `kyber.rules.exprs` would have run that package's `@rule` decorators
    from two families that do not currently import it, changing rule registration order.

    Args:
        expr: The expression to inspect.

    Returns:
        The integer value, or `None` when `expr` is not a plain integer literal.
    """
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


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


#: Scalar types an `InList` set may hold. Each has a `Lit` encoding in the JSON IR, so a set
#: built from them round-trips into `bc_expr::Expr::InList`'s `Vec<Literal>`.
_IN_LIST_SCALARS = (int, float, str, bytes, _dt.date, _dt.datetime)


def _in_list_foldable(values: list) -> bool:
    """Whether `values` can be an `InList` set rather than a chain of equalities.

    The exclusions mirror `kyber.rules.normalize.disjunctions`, which folds this same shape
    from the other direction and has to agree with this:

    * a non-scalar member (an `Expr`) has no `Literal` encoding at all;
    * a **bool**, whose set is not the one the engine would build beside an int;
    * a **NaN**, because `InList` probes a hash set and ``NaN != NaN``, while the engine's
      ``=`` *does* match a NaN row — so folding one would silently drop the rows the
      equality selects.

    A mixed-type set is refused for the same reason as the bool: the members have to be one
    type for the engine to build a typed set of them.
    """
    kinds = {type(v) for v in values}
    if len(kinds) != 1:
        return False
    (kind,) = kinds
    if kind is bool or kind not in _IN_LIST_SCALARS:
        return False
    return not any(is_nan(v) for v in values)


def _membership_test(input: Expr, values: list) -> Expr:
    """``input IN (values)`` over a non-empty list of non-null members.

    Prefers the `InList` node, which lowers to one hash-set probe per row
    (`bc_expr::eval::in_list`) rather than one full-column comparison per member, and which
    is the shape eight existing Kyber rules match on (`prune_in_list_by_zonemap`,
    `dedup_in_list`, `intersect_in_lists`, …). Building the chain and leaving the fold to
    `or_equalities_to_in_list` gave those rules nothing outside a `Filter`, and it cost more
    than plan quality: the chain is left-deep, one IR nesting level per member, and the
    engine's `serde` reader descends it recursively — so a *projection* over ~100 members
    overflowed the Rust stack and took the process down with SIGSEGV rather than raising.
    ``TfidfVectorizer(stop_words="english")`` is 318 members, which is how it was found.

    The fallback goes through `combine_disjuncts`, which builds a balanced tree, so a set
    this cannot fold — expression members, mixed types — is `log2(n)` levels deep instead of
    `n`. Imported inside the function because `plan.expr_rewrite` imports this module.
    """
    from batcher.plan.expr_rewrite import combine_disjuncts

    if _in_list_foldable(values):
        return InList(input, tuple(values))
    return combine_disjuncts([input == v for v in values])


def _null_safe_membership(input: Expr, values: list) -> Expr:
    """``is_in(values, nulls_equal=True)``: a two-valued membership where null equals ``None``.

    The SQL test answers null for a null input or a non-match against a list holding a
    null; coalescing it to false and OR-ing in ``input IS NULL`` (only when `values` holds
    ``None``) gives the null-safe reading with no new node.
    """
    non_null = [v for v in values if v is not None]
    hit: Expr = (
        Coalesce([_membership_test(input, non_null), Lit(False)]) if non_null else Lit(False)
    )
    if len(non_null) == len(values):
        return hit
    return hit | IsNull(input)


def _twos_complement_digits(value: Expr, radix: int, plain: Expr) -> Expr:
    """`value` in the power-of-two `radix`, a negative one as its 64-bit two's complement.

    A radix of ``2**k`` spends ``k`` bits a digit, so 64 bits are ``ceil(64 / k)`` digits
    whose top one holds the leftover high bits. The top digit is read with an arithmetic
    shift and a mask, the rest from the low bits masked non-negative and left-padded with
    zeros, so no step needs an unsigned 64-bit type the engine does not have.
    """
    from batcher.plan.expr_ir.constructors import when

    bits = radix.bit_length() - 1
    digits = -(-64 // bits)
    low_bits = (digits - 1) * bits
    top_bits = 64 - low_bits
    top = (value >> Lit(low_bits)).bitwise_and(Lit((1 << top_bits) - 1))
    low = value.bitwise_and(Lit((1 << low_bits) - 1))
    negative = Binary("concat", top.to_base(radix), low.to_base(radix).str.lpad(digits - 1, "0"))
    return when(value < Lit(0)).then(negative).otherwise(plain)


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


#: An explicit window frame as the expression layer spells it: signed `(start, end)`
#: offsets from the current row, optionally followed by the units they are counted in
#: (``"rows"``, the default, ``"range"``, or ``"groups"``). Negative is *preceding*, ``0``
#: is the current row, ``None`` is unbounded. `plan.logical.WindowFrame` is the validated
#: form this lowers to.
FrameSpec = Union[
    "tuple[int | None, int | None]",
    "tuple[int | None, int | None, str]",
]


def normalize_key_list(keys: IntoExpr | Iterable[IntoExpr] | None) -> list[IntoExpr]:
    """Normalize a ``partition_by``/``order_by`` argument to a list of key expressions.

    A single ``str`` column name or a lone ``Expr`` is wrapped in a one-element list; an
    existing iterable of keys is materialized with ``list``. Without this, the natural
    scalar spellings silently corrupt: ``over(partition_by="grp")`` would ``list("grp")``
    into ``['g', 'r', 'p']`` (partition by three phantom columns), and
    ``over(partition_by=col("g"))`` would iterate an `Expr` — which has an unbounded
    `__getitem__` — until memory is exhausted.

    ``None`` means *no keys* — an unpartitioned or unordered window, which is what SQL's
    bare ``OVER ()`` is and what a caller passing a conditionally-set variable holds.
    Without this case it reached ``list(None)`` and surfaced as a bare
    ``TypeError: 'NoneType' object is not iterable`` from inside the plan builder.
    """
    if keys is None:
        return []
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

    __slots__ = ("func", "input", "input2", "interpolation", "name", "param")

    def __init__(
        self,
        func: str,
        input: Expr | None,
        *,
        input2: Expr | None = None,
        param: float | None = None,
        name: str | None = None,
        interpolation: str | None = None,
    ) -> None:
        """Construct an aggregate over an optional input, plus an optional `input2` or `param`."""
        self.func = func
        self.input = input
        # The output column name set by `.alias(...)`, read by `group_by().agg()` when it
        # names a *positional* aggregate. It is consumed at the API surface and never
        # reaches `to_ir`, where the name is carried by `AggregateSpec.alias` instead --
        # which is why the Kyber rules that rebuild an `AggExpr` may drop it safely.
        self.name = name
        # The second input expression — the ordering key for arg_min/arg_max or the
        # paired column for corr/covar; None for unary and parametric aggregates.
        self.input2 = input2
        # The scalar parameter for parametric aggregates (the q of quantile); None otherwise.
        self.param = param
        # How `quantile` resolves a rank between two values (`plan.ir_tags.
        # QUANTILE_INTERPOLATIONS`). None is linear, and is omitted from the IR so a linear
        # quantile serializes as it always has. Every site that rebuilds an `AggExpr` from
        # another must carry it, or a `nearest` quantile silently turns linear.
        self.interpolation = interpolation

    def __repr__(self) -> str:
        """A source-like rendering, e.g. ``col('x').sum()`` or ``count()``."""
        args = []
        if self.input2 is not None:
            args.append(repr(self.input2))
        if self.param is not None:
            args.append(repr(self.param))
        if self.interpolation is not None:
            args.append(repr(self.interpolation))
        call = f"{self.func}({', '.join(args)})"
        rendered = call if self.input is None else f"{self.input!r}.{call}"
        return rendered if self.name is None else f"{rendered}.alias({self.name!r})"

    def alias(self, name: str) -> AggExpr:
        """Name this aggregate's output column — the Polars ``.alias(...)`` spelling.

        ``agg(total=col("x").sum())`` and ``agg(col("x").sum().alias("total"))`` build
        the same aggregate. The second is what a ported Polars or PySpark script is
        already written as, and it is the only positional spelling that can name a
        `count()` (which has no input column to be named after) or two aggregates over
        one column.

        Args:
            name: The output column name to bind this aggregate to.

        Returns:
            A new `AggExpr` naming its output `name`; this one is unchanged.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
                >>> ds.group_by("g").agg(
                ...     bt.col("x").sum().alias("total"), bt.count().alias("n")
                ... ).sort("g").to_pydict()
                {'g': ['a', 'b'], 'total': [3, 3], 'n': [2, 1]}
        """
        return AggExpr(
            self.func,
            self.input,
            input2=self.input2,
            param=self.param,
            name=name,
            interpolation=self.interpolation,
        )

    def to_ir(self, alias: str | None = None) -> dict[str, Any]:
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
        if alias is None:
            # Reached as a *child* of some `Expr`'s `to_ir()` — i.e. an aggregate used
            # where a scalar expression is expected. `group_by().agg()` splits aggregate
            # leaves out before lowering, so an unaliased call means it escaped that path.
            raise PlanError(
                "an aggregate expression (e.g. col('x').sum()) can only be used inside "
                "group_by().agg(); it cannot appear in select/with_columns/filter"
            )
        item: dict[str, Any] = {"func": self.func, "alias": alias}
        if self.input is not None:
            item["input"] = self.input.to_ir()
        if self.input2 is not None:
            item["input2"] = self.input2.to_ir()
        if self.param is not None:
            item["param"] = self.param
        if self.interpolation is not None:
            item["interpolation"] = self.interpolation
        return item

    def over(
        self,
        partition_by: Iterable[IntoExpr] | None = (),
        order_by: Iterable[IntoExpr] | None = (),
        frame: FrameSpec | None = None,
        *,
        descending: bool | Iterable[bool] = False,
        nulls_last: bool = True,
        mapping_strategy: str = "group_to_rows",
    ) -> WindowExpr:
        """Turn this aggregate into a window expression — SQL ``<agg> OVER (…)``.

        ``col("x").sum().over(partition_by=["g"])`` computes the per-partition sum
        broadcast to every row (no grouping/row collapse). With `order_by` it becomes
        a running aggregate; `frame` sets an explicit window. Used inside
        `with_columns`, which lowers it to the relational `Window` operator. The window
        aggregates (`sum`/`mean`/`min`/`max`/`count`, ...) and `first`/`last` support
        `over`; the two-input aggregates (`corr`, `covar_*`) do not.

        **The frame bounds are signed offsets, not PRECEDING/FOLLOWING magnitudes.**
        Negative precedes the current row, ``0`` is the current row, positive follows,
        and ``None`` is unbounded in that direction -- so SQL's
        ``ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`` is ``frame=(-2, 0)``, not
        ``(2, 0)``. Reading them as magnitudes is not a harmless slip: ``(2, 0)`` is
        rejected outright, and ``(2, 2)`` is *accepted* as "two following through two
        following" and quietly answers a different question than the one intended.

        An optional third element chooses the frame units, ``"rows"`` (the default),
        ``"range"`` or ``"groups"``, which differ exactly when rows tie on the
        ``order_by`` key: ``"rows"`` counts rows, ``"range"`` and ``"groups"`` treat a
        run of tied rows as one unit, matching SQL.

        Args:
            partition_by: Key expressions whose groups the aggregate is computed within.
                ``None`` or empty means unpartitioned, over the whole input.
            order_by: Expressions to order rows by, making it a running aggregate.
                ``None`` or empty leaves the aggregate unordered.
            frame: ``(start, end)`` signed offsets, optionally with a third units element
                as ``(start, end, units)``; ``None`` for the default frame.
            descending: Order every `order_by` key, or each one, largest first.
            nulls_last: Sort null keys after the non-null ones (the SQL default).
            mapping_strategy: How a group's result maps back to rows; only
                ``"group_to_rows"`` is supported.

        Returns:
            A window expression, one value per input row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 10]})
                >>> w = bt.col("v").sum().over(partition_by=["g"])
                >>> ds.with_columns(total=w).sort("v").to_pydict()
                {'g': ['a', 'a', 'b'], 'v': [1, 2, 10], 'total': [3, 3, 10]}

                >>> ds = bt.from_pydict({"t": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0]})
                >>> trailing = bt.col("v").sum().over(order_by=["t"], frame=(-2, 0))
                >>> ds.with_columns(s=trailing).sort("t").to_pydict()["s"]
                [1.0, 3.0, 6.0, 9.0]
        """
        from batcher.plan.expr_rewrite.over import bind_over

        return bind_over(  # type: ignore[return-value]
            self,
            partition_by,
            order_by,
            frame,
            descending=descending,
            nulls_last=nulls_last,
            mapping_strategy=mapping_strategy,
        )

    # --- arithmetic over aggregates ---------------------------------------
    # An aggregate can be combined with scalars and other aggregates into one
    # derived output — ``col("x").sum() / col("y").sum()``, ``corr(y, x) ** 2``.
    # The operators reuse `Expr`'s node-building implementations verbatim (so the
    # semantics are byte-identical), embedding this `AggExpr` as a leaf of the
    # resulting `Expr`. `group_by().agg()` then splits the leaves back out into the
    # aggregate pass and computes the surrounding expression in a following
    # projection — one mergeable aggregate, one stateless map, distributed-safe.

    def cast(self, dtype: str) -> Cast:
        """Cast this aggregate's result to an Arrow type named as a string.

        Lets an aggregate join an expression over aggregates at a chosen type — e.g.
        forcing an integer ``sum`` to Float64 before a division. The cast applies to the
        aggregated value, so it runs in the projection after the aggregate pass.

        Args:
            dtype: Target Arrow type name (e.g. ``"int64"``, ``"float64"``).

        Returns:
            An expression of the aggregate's result converted to `dtype`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.agg(r=bt.col("x").sum().cast("float64")).to_pydict()
                {'r': [6.0]}
        """
        return Cast(self, dtype)

    def __add__(self, other: IntoExpr) -> Expr:
        """Combine this aggregate with `other` by addition (``agg + other``)."""
        return Expr.__add__(self, other)

    def __radd__(self, other: IntoExpr) -> Expr:
        """Reflected addition so ``scalar + agg`` works."""
        return Expr.__radd__(self, other)

    def __sub__(self, other: IntoExpr) -> Expr:
        """Combine this aggregate with `other` by subtraction (``agg - other``)."""
        return Expr.__sub__(self, other)

    def __rsub__(self, other: IntoExpr) -> Expr:
        """Reflected subtraction so ``scalar - agg`` works."""
        return Expr.__rsub__(self, other)

    def __mul__(self, other: IntoExpr) -> Expr:
        """Combine this aggregate with `other` by multiplication (``agg * other``)."""
        return Expr.__mul__(self, other)

    def __rmul__(self, other: IntoExpr) -> Expr:
        """Reflected multiplication so ``scalar * agg`` works."""
        return Expr.__rmul__(self, other)

    def __truediv__(self, other: IntoExpr) -> Expr:
        """Divide this aggregate by `other` (``agg / other``, → Float64)."""
        return Expr.__truediv__(self, other)

    def __rtruediv__(self, other: IntoExpr) -> Expr:
        """Reflected true division so ``scalar / agg`` works (→ Float64)."""
        return Expr.__rtruediv__(self, other)

    def __floordiv__(self, other: IntoExpr) -> Expr:
        """Floor-divide this aggregate by `other` (``agg // other``)."""
        return Expr.__floordiv__(self, other)

    def __rfloordiv__(self, other: IntoExpr) -> Expr:
        """Reflected floor division so ``scalar // agg`` works."""
        return Expr.__rfloordiv__(self, other)

    def __mod__(self, other: IntoExpr) -> Expr:
        """Modulo of this aggregate by `other` (``agg % other``)."""
        return Expr.__mod__(self, other)

    def __rmod__(self, other: IntoExpr) -> Expr:
        """Reflected modulo so ``scalar % agg`` works."""
        return Expr.__rmod__(self, other)

    def __pow__(self, other: IntoExpr) -> Expr:
        """Raise this aggregate to `other` (``agg ** other``, → Float64)."""
        return Expr.__pow__(self, other)

    def __rpow__(self, other: IntoExpr) -> Expr:
        """Reflected exponentiation so ``scalar ** agg`` works (→ Float64)."""
        return Expr.__rpow__(self, other)

    def __neg__(self) -> Expr:
        """Arithmetic negation ``-agg``."""
        return Expr.__neg__(self)

    def __abs__(self) -> Expr:
        """Absolute value ``abs(agg)``."""
        return Expr.__abs__(self)

    # --- comparison and boolean composition over aggregates ---------------
    # These forward to `Expr` for the same reason the arithmetic ones do, and their
    # absence was not merely a missing feature. Without `__eq__`, Python fell back to
    # identity comparison, so ``col("x").sum() == 6`` evaluated to the *bool* `False`
    # rather than building a predicate — and `with_columns` then wrote that constant
    # into a column, silently reporting `False` for a sum that really was 6. Every
    # comparison is defined here so no such fallback remains.

    def __eq__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        """Equality predicate over this aggregate (``agg == other``)."""
        return Expr.__eq__(self, other)

    def __ne__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        """Inequality predicate over this aggregate (``agg != other``)."""
        return Expr.__ne__(self, other)

    def __lt__(self, other: IntoExpr) -> Expr:
        """Less-than predicate over this aggregate (``agg < other``)."""
        return Expr.__lt__(self, other)

    def __le__(self, other: IntoExpr) -> Expr:
        """Less-or-equal predicate over this aggregate (``agg <= other``)."""
        return Expr.__le__(self, other)

    def __gt__(self, other: IntoExpr) -> Expr:
        """Greater-than predicate over this aggregate (``agg > other``)."""
        return Expr.__gt__(self, other)

    def __ge__(self, other: IntoExpr) -> Expr:
        """Greater-or-equal predicate over this aggregate (``agg >= other``)."""
        return Expr.__ge__(self, other)

    def __and__(self, other: IntoExpr) -> Expr:
        """Boolean conjunction over aggregate predicates (``agg & other``)."""
        return Expr.__and__(self, other)

    def __rand__(self, other: IntoExpr) -> Expr:
        """Reflected conjunction so ``other & agg`` works."""
        return Expr.__rand__(self, other)

    def __or__(self, other: IntoExpr) -> Expr:
        """Boolean disjunction over aggregate predicates (``agg | other``)."""
        return Expr.__or__(self, other)

    def __ror__(self, other: IntoExpr) -> Expr:
        """Reflected disjunction so ``other | agg`` works."""
        return Expr.__ror__(self, other)

    def __invert__(self) -> Expr:
        """Boolean negation of an aggregate predicate (``~agg``)."""
        return Expr.__invert__(self)

    def __hash__(self) -> NoReturn:
        """Refuse hashing, exactly as `Expr` does — ``==`` now builds a predicate.

        Defining `__eq__` above would otherwise leave `AggExpr` with an inherited
        `__hash__` whose contract it no longer honors, so a set or dict keyed on
        aggregates would compare with a predicate and misbehave silently.

        Raises:
            TypeError: Always — naming the two workable keys.
        """
        return Expr.__hash__(self)


# Expose `Expr`'s unary/parametric math methods on `AggExpr` so an aggregate result can
# be transformed inside `group_by().agg()` — ``col("x").sum().sqrt()``,
# ``col("x").mean().round(2)``. Each embeds the aggregate as a leaf of the `Expr` the
# method builds; the aggregate-expression splitter then evaluates the transform in the
# projection after the aggregate pass. Bound by reference so the semantics and the
# documented examples are exactly `Expr`'s — one definition, one behavior.
for _agg_math_method in (
    "sqrt", "cbrt", "exp", "ln", "log2", "log10", "log1p", "expm1", "square",
    "abs", "sign", "round", "floor", "ceil", "trunc", "clip",
):  # fmt: skip
    setattr(AggExpr, _agg_math_method, getattr(Expr, _agg_math_method))
del _agg_math_method


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


# The pandas-compatible spellings (``isna``, ``fillna``, ``add``, …) are defined in
# `expr_ir.compat` and attached here, so this module stays the one-`Expr` hierarchy
# rather than carrying a second, parallel copy of its own surface. The import is at
# the bottom because `compat` names `Expr` only under `TYPE_CHECKING`.
