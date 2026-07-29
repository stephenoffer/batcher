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
import itertools
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NoReturn, Union

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.plan.expr_ir.compat import bind_compat_methods as _bind_compat_methods
from batcher.plan.expr_ir.compat import expr_attribute_error as _expr_attribute_error
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
    # `AggExpr` is not an `Expr` but can be a leaf of one (``col("x").sum() / 2``);
    # pass it through rather than lifting it to a `Lit`. `group_by().agg()` splits such
    # leaves back out; any that reach `to_ir()` elsewhere raise a clear error there.
    if isinstance(value, (Expr, AggExpr)):
        return value  # type: ignore[return-value]
    # An unterminated CASE builder is the one non-`Expr` that users hand us on purpose, by
    # forgetting `.otherwise(...)`. Lifting it to a literal produced `unsupported literal
    # type: CaseBuilder`, which names an internal class and no remedy. Catch it by name to
    # avoid importing `nodes` (which imports this module).
    if type(value).__name__ == "CaseBuilder":
        raise PlanError(
            "when(...).then(...) is an unfinished CASE builder, not an expression: it needs "
            "a terminating .otherwise(...). SQL's bare `CASE WHEN ... END` yields NULL, which "
            "has no literal spelling here — give .otherwise() an explicit sentinel and turn "
            "it into a null with bt.nullif(expr, sentinel) if that is what you want."
        )
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
        Float64 would silently lose precision. A zero divisor yields NULL (matching
        DuckDB and Polars) rather than ``inf``/``nan``. Float operands give the IEEE
        ``floor(a / b)``."""
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
            "For a row-wise minimum/maximum across columns use min_horizontal(a, b) / "
            "max_horizontal(a, b); for a column aggregate use .min() / .max()"
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
        # Type names are matched case-insensitively: pandas spells these `"Int64"` /
        # `"Float64"` and SQL `"BIGINT"`, and a case mismatch is a typo the user cannot
        # see. The IR always carries the canonical lowercase name, so the wire contract
        # is unaffected.
        canonical = dtype.lower() if isinstance(dtype, str) else dtype
        if canonical not in CAST_DTYPES:
            import difflib

            hint = difflib.get_close_matches(canonical, sorted(CAST_DTYPES), n=2, cutoff=0.5)
            suffix = f"; did you mean {' or '.join(map(repr, hint))}?" if hint else ""
            raise PlanError(f"unknown cast dtype {dtype!r}; valid: {sorted(CAST_DTYPES)}{suffix}")
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

    def to_base(self, radix: int) -> Expr:
        """This integer written in `radix` (DuckDB ``to_base``; ``bin`` is radix 2, → Utf8).

        Args:
            radix: The base, from 2 to 36.

        Returns:
            A new Utf8 expression: the digits, with a leading ``-`` when negative.

        Raises:
            PlanError: If `radix` is outside 2..36.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"n": [15, 255]})
                >>> ds.select(b=bt.col("n").to_base(2), h=bt.col("n").to_base(16)).to_pydict()
                {'b': ['1111', '11111111'], 'h': ['f', 'ff']}
        """
        from batcher.plan.expr_ir.func_nodes import StrFunc

        if not 2 <= radix <= 36:
            raise PlanError(f"to_base(): radix must be between 2 and 36, got {radix}")
        return StrFunc("to_base", self, start=radix)

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
                {'r': ['512 B', '1.5 KiB']}
        """
        from batcher.plan.expr_ir.func_nodes import StrFunc

        return StrFunc("format_bytes_si" if si else "format_bytes", self)

    def neg(self) -> Expr:
        """Arithmetic negation — the Polars ``neg`` spelling of the unary minus.

        Returns:
            A new expression of the negated values (nulls propagate).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, -2, 3]})
                >>> ds.select(r=bt.col("x").neg()).to_pydict()
                {'r': [-1, 2, -3]}
        """
        return Lit(0) - self

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

    def asinh(self) -> Expr:
        """Inverse hyperbolic sine (→ Float64; defined for all reals; nulls propagate).

        Composed as ``ln(x + sqrt(x*x + 1))``, matching NumPy/DuckDB ``asinh``.

        Returns:
            A new Float64 expression of the inverse hyperbolic sines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").asinh()).to_pydict()
                {'r': [0.0, 0.8813735870195429]}
        """
        return (self + (self * self + Lit(1)).sqrt()).ln()

    def acosh(self) -> Expr:
        """Inverse hyperbolic cosine (→ Float64; defined for ``x >= 1``; nulls propagate).

        Composed as ``ln(x + sqrt(x*x - 1))``, matching NumPy/DuckDB ``acosh``. Inputs
        below 1 yield NaN, as the real inverse is undefined there.

        Returns:
            A new Float64 expression of the inverse hyperbolic cosines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, 2.0]})
                >>> ds.select(r=bt.col("x").acosh()).to_pydict()
                {'r': [0.0, 1.3169578969248166]}
        """
        return (self + (self * self - Lit(1)).sqrt()).ln()

    def atanh(self) -> Expr:
        """Inverse hyperbolic tangent (→ Float64; defined for ``|x| < 1``; nulls propagate).

        Composed as ``0.5 * ln((1 + x) / (1 - x))``, matching NumPy/DuckDB ``atanh``.
        ``|x| >= 1`` yields ±inf/NaN, as the real inverse diverges there.

        Returns:
            A new Float64 expression of the inverse hyperbolic tangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 0.5]})
                >>> ds.select(r=bt.col("x").atanh()).to_pydict()
                {'r': [0.0, 0.5493061443340549]}
        """
        return Lit(0.5) * ((Lit(1) + self) / (Lit(1) - self)).ln()

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
        return self.asin()

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
        return self.acos()

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
        return self.atan()

    def arcsinh(self) -> Expr:
        """Inverse hyperbolic sine — the Polars/NumPy ``arcsinh`` spelling of :meth:`asinh`.

        Returns:
            A new Float64 expression of the inverse hyperbolic sines.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
                >>> ds.select(r=bt.col("x").arcsinh()).to_pydict()
                {'r': [0.0, 0.8813735870195429]}
        """
        return self.asinh()

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
        return self.acosh()

    def arctanh(self) -> Expr:
        """Inverse hyperbolic tangent — the Polars/NumPy ``arctanh`` spelling of :meth:`atanh`.

        Returns:
            A new Float64 expression of the inverse hyperbolic tangents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.0, 0.5]})
                >>> ds.select(r=bt.col("x").arctanh()).to_pydict()
                {'r': [0.0, 0.5493061443340549]}
        """
        return self.atanh()

    def is_between(self, lower: IntoExpr, upper: IntoExpr, closed: str = "both") -> Expr:
        """Range test — the Polars ``is_between`` spelling of :meth:`between`.

        Args:
            lower: The lower bound.
            upper: The upper bound.
            closed: Which bounds are inclusive — ``"both"``/``"left"``/``"right"``/``"none"``.

        Returns:
            A boolean expression, true where the value lies in the range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.select(r=bt.col("x").is_between(2, 3)).to_pydict()
                {'r': [False, True, True]}
        """
        return self.between(lower, upper, closed)

    def clip_min(self, lower: IntoExpr) -> Expr:
        """Clamp values up to at least `lower` — the Polars ``clip_min`` spelling of :meth:`clip`.

        Args:
            lower: The lower bound; values below it become it.

        Returns:
            A new expression with each value raised to at least `lower`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-1, 5, 20]})
                >>> ds.select(r=bt.col("x").clip_min(0)).to_pydict()
                {'r': [0, 5, 20]}
        """
        return self.clip(lower=lower)

    def clip_max(self, upper: IntoExpr) -> Expr:
        """Clamp values down to at most `upper` — the Polars ``clip_max`` spelling of :meth:`clip`.

        Args:
            upper: The upper bound; values above it become it.

        Returns:
            A new expression with each value lowered to at most `upper`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [-1, 5, 20]})
                >>> ds.select(r=bt.col("x").clip_max(10)).to_pydict()
                {'r': [-1, 5, 10]}
        """
        return self.clip(upper=upper)

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
                >>> ds.with_columns(m=bt.col("x").expanding_mean()).to_pydict()["m"]
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
                >>> ds.with_columns(v=bt.col("x").expanding_var().round(4)).to_pydict()["v"]
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
                >>> ds.with_columns(s=bt.col("x").expanding_std().round(4)).to_pydict()["s"]
                [nan, 0.7071, 1.0, 1.291]
        """
        return self.expanding_var(partition_by, order_by, ddof).sqrt()

    def hash_bucket(self, buckets: int, seed: int = 0) -> Expr:
        """Assign each value to one of `buckets` by a stable hash — ``|hash(x)| % buckets``.

        Deterministic across partitions, runs, and machines, which is what makes it a
        safe key for a reproducible train/test split, a shard assignment, or an A/B
        bucket.

        Args:
            buckets: How many buckets to spread values across (must be >= 1).
            seed: Hash seed; vary it for an independent bucketing of the same keys.

        Returns:
            An Int64 expression in ``[0, buckets)``.

        Raises:
            PlanError: If `buckets` < 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"k": ["a", "b", "c", "d"]})
                >>> ds.select(b=bt.col("k").hash_bucket(4)).to_pydict()
                {'b': [1, 3, 3, 1]}
        """
        buckets = require_int(buckets, func="hash_bucket", arg="buckets", minimum=1)
        seed = require_int(seed, func="hash_bucket", arg="seed")
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
                >>> ds.with_columns(c=bt.col("x").cumulative_pct()).to_pydict()["c"]
                [0.1, 0.3, 0.6, 1.0]
        """
        keys = list(partition_by)
        running = self.cum_sum(partition_by=keys, order_by=list(order_by))
        return running / self.sum().over(partition_by=keys)

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

    def is_first_distinct(self, order_by: IntoExpr) -> Expr:
        """True on the first occurrence of each distinct value, in `order_by` order.

        The de-duplication marker: filtering on it keeps one row per distinct value.
        `order_by` is required so the choice is deterministic and partition-independent
        (an arrival-order "first" would differ between a single-node and a distributed
        run).

        Args:
            order_by: The expression whose ascending order decides which occurrence
                counts as first.

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

        rn = row_number().over(partition_by=[self], order_by=[_col_or_expr(order_by)])
        return rn == Lit(1)

    def is_last_distinct(self, order_by: IntoExpr) -> Expr:
        """True on the last occurrence of each distinct value, in `order_by` order.

        The mirror of :meth:`is_first_distinct`, useful for keeping the most recent row
        per key. `order_by` is likewise required for determinism.

        Args:
            order_by: The expression whose ascending order decides which occurrence
                counts as last.

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

        key = _col_or_expr(order_by)
        rn = row_number().over(partition_by=[self], order_by=[key])
        total = self.count().over(partition_by=[self])
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

    def rint(self) -> MathExpr:
        """Round half to **even** — IEEE-754 ``roundTiesToEven`` (→ Float64).

        The tie rule is the difference from :meth:`round`, which rounds half *away from
        zero* here and in DuckDB: ``rint(2.5)`` is ``2.0`` where ``round(2.5)`` is
        ``3.0``. Ties-to-even is what floating-point arithmetic itself uses, so summing
        rounded values does not drift upward the way half-up rounding does.

        Returns:
            A new Float64 expression of the rounded values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [0.5, 1.5, 2.5, 3.5, -2.5]})
                >>> ds.select(r=bt.col("x").rint()).to_pydict()
                {'r': [0.0, 2.0, 2.0, 4.0, -2.0]}
        """
        return MathExpr("rint", self)

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
        """``n!`` — factorial of a non-negative integer (DuckDB ``factorial``; → Float64).

        Returns:
            A new Float64 expression of the factorials.

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
                {'g': ['a'], 'r': [3.152000000000001]}
        """
        return AggExpr("kurtosis", self)

    def kurtosis_pop(self) -> AggExpr:
        """Population excess kurtosis per group (→ Float64).

        The uncorrected ``m4/m2² - 3``, where :meth:`kurtosis` applies the sample
        correction. DuckDB has both, and on a small group they differ by a lot, so
        pick the one your statistics call for rather than treating them as rounding.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 5, "x": [1, 2, 3, 4, 10]})
                >>> ds.group_by("g").agg(r=bt.col("x").kurtosis_pop()).to_pydict()
                {'g': ['a'], 'r': [0.5049148147577234]}
        """
        return AggExpr("kurtosis_pop", self)

    def entropy(self) -> AggExpr:
        """Base-2 Shannon entropy of a group's value distribution (→ Float64).

        ``-Σ pᵢ·log₂(pᵢ)`` over the distinct values' frequencies: 0 when a group holds
        one distinct value, ``log₂(n)`` when all n are distinct. The measure to reach
        for when the question is how *concentrated* a column is, rather than how large.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 4, "x": [1, 1, 2, 2]})
                >>> ds.group_by("g").agg(r=bt.col("x").entropy()).to_pydict()
                {'g': ['a'], 'r': [1.0]}
        """
        return AggExpr("entropy", self)

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
        return AggExpr("quantile_disc", self, param=q)

    def top_k(self, k: int) -> AggExpr:
        """The `k` most frequent values per group, most frequent first (→ List).

        DuckDB's ``approx_top_k``, computed **exactly**: the aggregate already holds
        every value of the group, so a sketch could only lose accuracy. Ties break to
        the smaller value, so the result does not depend on partition order.

        Args:
            k: How many values to return.

        Returns:
            An aggregate expression for use in ``group_by().agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a"] * 6, "x": [1, 2, 2, 3, 3, 3]})
                >>> ds.group_by("g").agg(r=bt.col("x").top_k(2)).to_pydict()
                {'g': ['a'], 'r': [[3, 2]]}
        """
        return AggExpr("approx_top_k", self, param=float(k))

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

        q = require_float(q, func="quantile", arg="q")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile q must be in [0, 1], got {q}")
        return AggExpr("quantile", self, param=q)

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

    def _rolling_var(
        self,
        window_size: int,
        ddof: int,
        min_periods: int | None,
        partition_by: Iterable[IntoExpr],
        order_by: Iterable[IntoExpr],
    ) -> Expr:
        """Sample/population variance over the trailing frame, composed from moments.

        ``Var = E[x^2] - E[x]^2`` gives the population variance over the frame; the
        Bessel correction ``n / (n - ddof)`` lifts it to the sample statistic. All the
        terms reuse the tested `_rolling` machinery over the *same* frame, so rolling
        variance adds no new IR and inherits the leading-partial-frame / `min_periods`
        semantics of :meth:`rolling_sum`.

        The values are **centered on the partition mean first**, which the identity
        ``Var(x) = Var(x - k)`` makes exact for any constant `k`. Without it this is the
        sum-of-powers formula that `bc-runtime`'s `var_state` was rewritten to escape: it
        subtracts two nearly equal large numbers, so it loses a digit of precision for
        every digit by which the mean exceeds the spread. Measured on
        ``[k+1, k+2, ..., k+6]`` with a 3-wide frame, where the true variance is 1.0:

        ==============  ==================================
        offset ``k``    ``E[x^2] - E[x]^2`` returned
        ==============  ==================================
        ``0``           1.0
        ``1e6``         0.999939
        ``1e9``         0.0        (reads as "constant")
        ``1e12``        -201326592 (a negative variance)
        ==============  ==================================

        An epoch-second timestamp is ~1.7e9 and a monetary column in cents reaches 1e12,
        so this is the ordinary case rather than an adversarial one. Centering removes the
        offset before it can cancel; the partition mean is used because it is the constant
        nearest the data that a window expression can name.

        The residual is clamped at zero for the rounding that can still put a
        mathematically non-negative quantity a few ulps below it — via a comparison, so a
        genuine NaN (a non-finite value in the frame) propagates instead of being clipped
        to a confident zero."""
        from batcher.plan.expr_ir.constructors import when

        centre = AggExpr("avg", self).over(partition_by=partition_by)
        centered = self - centre
        mean = centered._rolling("avg", window_size, min_periods, partition_by, order_by)
        mean_sq = (centered * centered)._rolling(
            "avg", window_size, min_periods, partition_by, order_by
        )
        raw = mean_sq - mean * mean
        var_pop = when(raw < Lit(0.0)).then(Lit(0.0)).otherwise(raw)
        if ddof == 0:
            return var_pop
        count = self._rolling("count", window_size, min_periods, partition_by, order_by).cast(
            "float64"
        )
        return var_pop * (count / (count - Lit(ddof)))

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
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                >>> ds.with_columns(v=bt.col("x").rolling_var(2, min_periods=2)).to_pydict()
                {'x': [1.0, 2.0, 3.0, 4.0], 'v': [None, 0.5, 0.5, 0.5]}
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
                >>> ds = bt.from_pydict({"x": [2.0, 4.0, 6.0]})
                >>> ds.with_columns(s=bt.col("x").rolling_std(2, min_periods=2)).to_pydict()
                {'x': [2.0, 4.0, 6.0], 's': [None, 1.4142135623730951, 1.4142135623730951]}
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

    # `_ir_cache` mirrors the memo `IRNode` keeps in its instance `__dict__`. `Lit` is
    # `__slots__`-based (there are more literals in a plan than any other node kind, and
    # a per-instance dict on each is real memory), so it needs the slot declared to get
    # the same one-lowering-per-node behavior every other node already has.
    __slots__ = ("_ir_cache", "value")

    def __init__(self, value: int | float | bool | str) -> None:
        """Wrap a Python scalar (or date/datetime) as a literal expression node."""
        self.value = value
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
        # (bool before int, datetime before date). A subclass of one of these — a
        # `numpy.bool_`, an `IntEnum` — still falls through to the ladder below and is
        # tagged exactly as it was.
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
                _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
                if v.tzinfo is not None
                else _dt.datetime(1970, 1, 1)
            )
            delta = v - epoch
            micros = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
            tagged = {"timestamp": micros}
        elif isinstance(v, _dt.date):
            tagged = {"date": (v - _dt.date(1970, 1, 1)).days}
        else:  # pragma: no cover - guarded by typing
            raise TypeError(f"unsupported literal type: {type(v).__name__}")
        out = {"e": ExprTag.LIT, "value": tagged}
        self._ir_cache = out
        return out


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
    "abs", "sign", "round", "pow", "floor", "ceil", "trunc", "clip",
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
_bind_compat_methods(Expr)
