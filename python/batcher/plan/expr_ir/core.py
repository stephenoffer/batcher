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

if TYPE_CHECKING:
    from batcher.plan.expr_ir.image import _ImageNamespace
    from batcher.plan.expr_ir.namespaces import (
        _DtNamespace,
        _JsonNamespace,
        _ListNamespace,
        _StrNamespace,
        _StructNamespace,
    )

# A value that can be promoted to an expression: another Expr or a Python scalar.
IntoExpr = Union["Expr", int, float, bool, str]

# The Arrow type names `cast` accepts, mirroring the engine's `parse_dtype`
# (bc-expr). Kept here so a bad dtype fails at plan-build time with a clear message
# instead of surfacing as an opaque error from the Rust FFI mid-execution.
CAST_DTYPES: frozenset[str] = frozenset(
    {
        "int64",
        "long",
        "int32",
        "int",
        "float64",
        "double",
        "float32",
        "float",
        "bool",
        "boolean",
        "string",
        "utf8",
        "date",
        "date32",
        "timestamp",
        "datetime",
    }
)


def _wrap(value: IntoExpr) -> Expr:
    return value if isinstance(value, Expr) else Lit(value)


class Expr:
    """Base class for scalar expressions. Subclasses are immutable IR nodes."""

    # --- serialization -----------------------------------------------------
    def to_ir(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    # --- comparison operators (yield boolean expressions) ------------------
    def __gt__(self, other: IntoExpr) -> Expr:
        return Binary("gt", self, _wrap(other))

    def __ge__(self, other: IntoExpr) -> Expr:
        return Binary("ge", self, _wrap(other))

    def __lt__(self, other: IntoExpr) -> Expr:
        return Binary("lt", self, _wrap(other))

    def __le__(self, other: IntoExpr) -> Expr:
        return Binary("le", self, _wrap(other))

    def __eq__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        return Binary("eq", self, _wrap(other))

    def __ne__(self, other: IntoExpr) -> Expr:  # type: ignore[override]
        return Binary("ne", self, _wrap(other))

    # Expr is used for plan building, not as a dict key; make that explicit.
    __hash__ = None  # type: ignore[assignment]

    # --- arithmetic operators ---------------------------------------------
    def __add__(self, other: IntoExpr) -> Expr:
        return Binary("add", self, _wrap(other))

    def __sub__(self, other: IntoExpr) -> Expr:
        return Binary("sub", self, _wrap(other))

    def __mul__(self, other: IntoExpr) -> Expr:
        return Binary("mul", self, _wrap(other))

    def __truediv__(self, other: IntoExpr) -> Expr:
        return Binary("div", self, _wrap(other))

    def __mod__(self, other: IntoExpr) -> Expr:
        return Binary("mod", self, _wrap(other))

    # reflected forms so `2 * col("x")` works
    def __radd__(self, other: IntoExpr) -> Expr:
        return Binary("add", _wrap(other), self)

    def __rsub__(self, other: IntoExpr) -> Expr:
        return Binary("sub", _wrap(other), self)

    def __rmul__(self, other: IntoExpr) -> Expr:
        return Binary("mul", _wrap(other), self)

    def __rtruediv__(self, other: IntoExpr) -> Expr:
        return Binary("div", _wrap(other), self)

    def __rmod__(self, other: IntoExpr) -> Expr:
        return Binary("mod", _wrap(other), self)

    # --- boolean operators (bitwise spelling, like Polars/pandas) ----------
    def __and__(self, other: IntoExpr) -> Expr:
        return Binary("and", self, _wrap(other))

    def __or__(self, other: IntoExpr) -> Expr:
        return Binary("or", self, _wrap(other))

    def __invert__(self) -> Expr:
        return Not(self)

    # --- bitwise integer operators (distinct from the boolean `&`/`|`) ------
    def bitwise_and(self, other: IntoExpr) -> Expr:
        """Bitwise AND of two integer expressions (operands cast to Int64)."""
        return Binary("bit_and", self, _wrap(other))

    def bitwise_or(self, other: IntoExpr) -> Expr:
        """Bitwise OR of two integer expressions."""
        return Binary("bit_or", self, _wrap(other))

    def bitwise_xor(self, other: IntoExpr) -> Expr:
        """Bitwise XOR of two integer expressions."""
        return Binary("bit_xor", self, _wrap(other))

    def bitwise_left_shift(self, other: IntoExpr) -> Expr:
        """Left-shift this integer expression by `other` bits."""
        return Binary("shift_left", self, _wrap(other))

    def bitwise_right_shift(self, other: IntoExpr) -> Expr:
        """Right-shift this integer expression by `other` bits."""
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
        """
        return Aliased(self, name)

    # --- unary / type methods ----------------------------------------------
    def cast(self, dtype: str) -> Cast:
        """Cast to an Arrow type by name (int64/float64/int32/bool/string/...).

        The dtype is validated at plan-build time; an unknown name raises rather than
        failing opaquely in the engine mid-query. A value that cannot be converted
        errors the query (DuckDB ``CAST``); use `try_cast` to get NULL instead.
        """
        return self._cast(dtype, try_cast=False)

    def try_cast(self, dtype: str) -> Cast:
        """Cast to an Arrow type by name, yielding NULL for values that cannot be
        converted (DuckDB ``TRY_CAST``) instead of erroring the query.

        The common safe-ingest spelling: ``col("x").try_cast("int64")`` turns a
        dirty string column into integers, with unparseable values becoming NULL
        (ready to `drop_nulls` or route to a quarantine sink).
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
        return IsNull(self)

    def is_not_null(self) -> IsNotNull:
        return IsNotNull(self)

    def is_in(self, values: Iterable[IntoExpr]) -> Expr:
        """``self IN (values)`` — true if equal to any value.

        Desugars to an OR of equality checks, so it follows SQL three-valued
        logic (``NULL IN (...)`` is NULL) and an empty collection is always false.
        """
        vals = list(values)
        if not vals:
            return Lit(False)
        expr: Expr = self == vals[0]
        for v in vals[1:]:
            expr = expr | (self == v)
        return expr

    def between(self, low: IntoExpr, high: IntoExpr) -> Expr:
        """``self BETWEEN low AND high`` (inclusive), matching SQL/DuckDB."""
        return (self >= low) & (self <= high)

    def eq_missing(self, other: IntoExpr) -> Expr:
        """Null-safe equality (SQL ``IS NOT DISTINCT FROM``): two nulls compare
        equal, and a null vs a non-null compares **false** (never null).

        The reliable way to compare possibly-null keys — used for change detection
        in slowly-changing dimensions. Desugars to existing ops (no new IR).
        """
        o = _wrap(other)
        both_null = self.is_null() & o.is_null()
        return Coalesce([self == o, Lit(False)]) | both_null

    def replace(self, mapping: dict[Any, Any], *, default: IntoExpr | None = None) -> Expr:
        """Remap values via a ``{old: new}`` dictionary (value standardization /
        lookup recode). Values absent from `mapping` keep their original value, or
        take `default` when one is given. Desugars to a ``CASE`` chain (no new IR).

        ``col("c").replace({"US": "USA", "UK": "GBR"})`` standardizes country codes.
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
        """String functions: `col("s").str.upper()`, `.str.contains("x")`, ..."""
        from batcher.plan.expr_ir.namespaces import _StrNamespace

        return _StrNamespace(self)

    @property
    def dt(self) -> _DtNamespace:
        """Date/time field extraction: `col("d").dt.year()`, `.dt.month()`, ..."""
        from batcher.plan.expr_ir.namespaces import _DtNamespace

        return _DtNamespace(self)

    # --- math functions ----------------------------------------------------
    def abs(self) -> MathExpr:
        return MathExpr("abs", self)

    def round(self, digits: int | None = None) -> Expr:
        """Round to the nearest integer, or to ``digits`` decimal places."""
        if digits is None:
            return MathExpr("round", self)
        return Math2Expr("round", self, Lit(digits))

    def pow(self, exponent: IntoExpr) -> Math2Expr:
        """This value raised to ``exponent`` (→ Float64)."""
        return Math2Expr("pow", self, _wrap(exponent))

    def __pow__(self, other: IntoExpr) -> Math2Expr:
        return Math2Expr("pow", self, _wrap(other))

    def __rpow__(self, other: IntoExpr) -> Math2Expr:
        return Math2Expr("pow", _wrap(other), self)

    def floor(self) -> MathExpr:
        return MathExpr("floor", self)

    def ceil(self) -> MathExpr:
        return MathExpr("ceil", self)

    def sqrt(self) -> MathExpr:
        return MathExpr("sqrt", self)

    def ln(self) -> MathExpr:
        """Natural logarithm (→ Float64)."""
        return MathExpr("ln", self)

    def log10(self) -> MathExpr:
        """Base-10 logarithm (→ Float64)."""
        return MathExpr("log10", self)

    def log2(self) -> MathExpr:
        """Base-2 logarithm (→ Float64)."""
        return MathExpr("log2", self)

    def exp(self) -> MathExpr:
        """e raised to this value (→ Float64)."""
        return MathExpr("exp", self)

    def sin(self) -> MathExpr:
        return MathExpr("sin", self)

    def cos(self) -> MathExpr:
        return MathExpr("cos", self)

    def tan(self) -> MathExpr:
        return MathExpr("tan", self)

    def sign(self) -> MathExpr:
        """-1, 0, or 1 by sign (→ Float64)."""
        return MathExpr("sign", self)

    def trunc(self) -> MathExpr:
        """Truncate toward zero (→ Float64)."""
        return MathExpr("trunc", self)

    def cbrt(self) -> MathExpr:
        """Cube root (→ Float64)."""
        return MathExpr("cbrt", self)

    def asin(self) -> MathExpr:
        return MathExpr("asin", self)

    def acos(self) -> MathExpr:
        return MathExpr("acos", self)

    def atan(self) -> MathExpr:
        return MathExpr("atan", self)

    def sinh(self) -> MathExpr:
        return MathExpr("sinh", self)

    def cosh(self) -> MathExpr:
        return MathExpr("cosh", self)

    def tanh(self) -> MathExpr:
        return MathExpr("tanh", self)

    def degrees(self) -> MathExpr:
        """Radians → degrees (→ Float64)."""
        return MathExpr("degrees", self)

    def radians(self) -> MathExpr:
        """Degrees → radians (→ Float64)."""
        return MathExpr("radians", self)

    def cot(self) -> MathExpr:
        """Cotangent, 1/tan (→ Float64)."""
        return MathExpr("cot", self)

    @property
    def list(self) -> _ListNamespace:
        """List/array reductions: `col("a").list.len()`, `.list.sum()`, …"""
        from batcher.plan.expr_ir.namespaces import _ListNamespace

        return _ListNamespace(self)

    # `.arr` is an alias for `.list` (Polars spelling).
    @property
    def arr(self) -> _ListNamespace:
        from batcher.plan.expr_ir.namespaces import _ListNamespace

        return _ListNamespace(self)

    @property
    def struct(self) -> _StructNamespace:
        """Struct field access: `col("s").struct.field("x")`."""
        from batcher.plan.expr_ir.namespaces import _StructNamespace

        return _StructNamespace(self)

    @property
    def json(self) -> _JsonNamespace:
        """JSON access on a string column: `col("j").json.extract_string("$.a")`."""
        from batcher.plan.expr_ir.namespaces import _JsonNamespace

        return _JsonNamespace(self)

    @property
    def image(self) -> _ImageNamespace:
        """Lazy image decode: `col("bytes").image.decode()` / `.image.to_tensor(224, 224)`."""
        from batcher.plan.expr_ir.image import _ImageNamespace

        return _ImageNamespace(self)

    def fill_null(self, value: IntoExpr) -> Coalesce:
        """Replace nulls with `value` (COALESCE of two)."""
        return Coalesce([self, _wrap(value)])

    # --- NaN handling / clamping -------------------------------------------
    def is_nan(self) -> Expr:
        """True where the value is IEEE NaN (a float-only notion, distinct from null).

        Nulls propagate (a null input yields null, not true). This is a dedicated op,
        not the ``self != self`` trick: the engine's ``!=`` uses total ordering
        (where ``NaN == NaN``), so ``self != self`` would never flag a NaN.
        """
        return IsNan(self)

    def is_not_nan(self) -> Expr:
        """True where the value is not NaN (the negation of `is_nan`; null → null)."""
        return Not(IsNan(self))

    def clip(self, lower: IntoExpr | None = None, upper: IntoExpr | None = None) -> Expr:
        """Clamp values into ``[lower, upper]`` (either bound optional).

        Nulls are preserved (a null stays null, not pulled to a bound): the lowering
        is a conditional, so a comparison against a null input is null and falls
        through to the original value.
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
        return AggExpr("sum", self)

    def min(self) -> AggExpr:
        return AggExpr("min", self)

    def max(self) -> AggExpr:
        return AggExpr("max", self)

    def mean(self) -> AggExpr:
        return AggExpr("mean", self)

    def var(self) -> AggExpr:
        """Sample variance (Bessel-corrected)."""
        return AggExpr("var", self)

    def std(self) -> AggExpr:
        """Sample standard deviation."""
        return AggExpr("stddev", self)

    def median(self) -> AggExpr:
        """Median (exact; averages the two middle values for an even count)."""
        return AggExpr("median", self)

    def quantile(self, q: float) -> AggExpr:
        """Continuous quantile at ``q`` in [0, 1] (linear interpolation).

        ``quantile(0.5)`` equals :meth:`median`. Raises ``PlanError`` if ``q`` is
        outside [0, 1].
        """
        from batcher._internal.errors import PlanError

        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile q must be in [0, 1], got {q}")
        return AggExpr("quantile", self, param=float(q))

    def count(self) -> AggExpr:
        """COUNT of non-null values of this expression."""
        return AggExpr("count", self)

    def n_unique(self) -> AggExpr:
        """COUNT(DISTINCT) — number of distinct non-null values of this expression."""
        return AggExpr("count_distinct", self)

    # SQL spelling; same aggregate as `n_unique`.
    count_distinct = n_unique

    def approx_n_unique(self) -> AggExpr:
        """Approximate COUNT(DISTINCT) via a HyperLogLog sketch (~2% error).

        Bounded memory regardless of skew — the skew-safe choice when an exact
        `n_unique` on a hot key would hold every distinct value. Mergeable, so it
        is identical single-node and distributed."""
        return AggExpr("approx_count_distinct", self)

    # SQL spelling; same aggregate as `approx_n_unique`.
    approx_count_distinct = approx_n_unique

    def approx_quantile(self, q: float) -> AggExpr:
        """Approximate quantile `q ∈ [0, 1]` via a KLL sketch (bounded memory).

        The skew-safe choice when an exact `quantile`/`median` on a hot key would
        hold every value. Mergeable, so identical single-node and distributed."""
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"approx_quantile(q) requires q in [0, 1], got {q}")
        return AggExpr("approx_quantile", self, param=float(q))

    def approx_median(self) -> AggExpr:
        """Approximate median (the 0.5 quantile) via a KLL sketch — see
        `approx_quantile`."""
        return AggExpr("approx_quantile", self, param=0.5)

    def mode(self) -> AggExpr:
        """Most frequent value per group. Ties are broken by the smallest value
        (deterministic and partition-independent). Works on any column type."""
        return AggExpr("mode", self)

    def first(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the first row in `order_by` order (SQL
        ``first(x ORDER BY order_by)``). Equivalent to ``arg_min(order_by)``.

        An explicit `order_by` is **required**: an arrival-order first/last is not
        partition-independent, so it could not stay identical single-node and
        distributed. With an order key the result is deterministic and mergeable
        (ties on the key break to the smallest value)."""
        return AggExpr("arg_min", self, input2=_wrap(order_by))

    def last(self, order_by: IntoExpr) -> AggExpr:
        """This expression's value at the last row in `order_by` order (SQL
        ``last(x ORDER BY order_by)``). Equivalent to ``arg_max(order_by)``. As with
        :meth:`first`, an explicit `order_by` is required so the result stays
        deterministic and mergeable across partitions."""
        return AggExpr("arg_max", self, input2=_wrap(order_by))

    def arg_min(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is minimal in the group
        (SQL ``arg_min``/``min_by``). Key ties break to the smallest value, so the
        result is deterministic and partition-independent."""
        return AggExpr("arg_min", self, input2=_wrap(by))

    def arg_max(self, by: IntoExpr) -> AggExpr:
        """This expression's value at the row where `by` is maximal in the group
        (SQL ``arg_max``/``max_by``)."""
        return AggExpr("arg_max", self, input2=_wrap(by))

    def bool_and(self) -> AggExpr:
        """Logical AND of this boolean expression's non-null values per group
        (null when the group has no non-null value)."""
        return AggExpr("bool_and", self)

    def bool_or(self) -> AggExpr:
        """Logical OR of this boolean expression's non-null values per group
        (null when the group has no non-null value)."""
        return AggExpr("bool_or", self)

    def array_agg(self) -> AggExpr:
        """Collect this expression's non-null values in each group into a ``List``
        (SQL ``array_agg``; Spark ``collect_list``). Without an explicit order the
        element order is arrival-dependent. Mergeable — the per-group value list is
        the partial state, so the result is the same single-node and distributed.

        Chain a list reduction on the result column to summarize it, e.g.
        ``ds.group_by("g").agg(tags=col("t").array_agg())`` then
        ``col("tags").list.join(",")``."""
        return AggExpr("list_agg", self)


class Lit(Expr):
    """A constant literal. The wire kind is inferred from the Python type."""

    __slots__ = ("value",)

    def __init__(self, value: int | float | bool | str) -> None:
        self.value = value

    def to_ir(self) -> dict[str, Any]:
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


class Binary(Expr):
    """A binary operation over two sub-expressions."""

    __slots__ = ("left", "op", "right")

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.BINARY,
            "op": self.op,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }


class Not(Expr):
    """Logical negation of a boolean sub-expression."""

    __slots__ = ("input",)

    def __init__(self, input: Expr) -> None:
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.NOT, "input": self.input.to_ir()}


class Cast(Expr):
    """Cast a sub-expression to a named Arrow type.

    `try_cast` selects DuckDB ``TRY_CAST`` semantics — a value that cannot be
    converted yields NULL instead of erroring the query; the default strict
    ``CAST`` errors on an invalid value.
    """

    __slots__ = ("dtype", "input", "try_cast")

    def __init__(self, input: Expr, dtype: str, *, try_cast: bool = False) -> None:
        self.input = input
        self.dtype = dtype
        self.try_cast = try_cast

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.CAST,
            "input": self.input.to_ir(),
            "dtype": self.dtype,
            "try_cast": self.try_cast,
        }


class IsNull(Expr):
    """True where the argument is null."""

    __slots__ = ("input",)

    def __init__(self, input: Expr) -> None:
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.IS_NULL, "input": self.input.to_ir()}


class IsNotNull(Expr):
    """True where the argument is non-null."""

    __slots__ = ("input",)

    def __init__(self, input: Expr) -> None:
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.IS_NOT_NULL, "input": self.input.to_ir()}


class IsNan(Expr):
    """True where a float value is IEEE NaN (null → null)."""

    __slots__ = ("input",)

    def __init__(self, input: Expr) -> None:
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.IS_NAN, "input": self.input.to_ir()}


class Aliased(Expr):
    """An expression tagged with an output name (from `Expr.alias`).

    Transparent in the IR — `to_ir` delegates to the wrapped expression, so the
    name is carried only at the API/projection boundary. Reachable via
    `Expr.alias(name)`; not constructed directly.
    """

    __slots__ = ("inner", "name")

    def __init__(self, inner: Expr, name: str) -> None:
        self.inner = inner
        self.name = name

    def to_ir(self) -> dict[str, Any]:
        return self.inner.to_ir()


class AggExpr:
    """An aggregate over an optional input expression.

    Built via `col(...).sum()` etc. or the top-level `count()`; bound to an output
    name when passed to `group_by(...).agg(name=agg)`. Serializes to the engine's
    `AggregateItem` shape.
    """

    __slots__ = ("func", "input", "input2", "param")

    def __init__(
        self,
        func: str,
        input: Expr | None,
        param: float | None = None,
        input2: Expr | None = None,
    ) -> None:
        self.func = func
        self.input = input
        # The second argument — the ordering key for arg_min/arg_max; None otherwise.
        self.input2 = input2
        self.param = param

    def to_ir(self, alias: str) -> dict[str, Any]:
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
        """
        from batcher.plan.expr_ir.nodes import WindowExpr

        # `mean` is the DataFrame spelling; the window engine names the aggregate `avg`.
        func = "avg" if self.func == "mean" else self.func
        return WindowExpr(func, self.input, list(partition_by), list(order_by), frame)


class MathExpr(Expr):
    """A unary math function over a numeric sub-expression."""

    __slots__ = ("fn", "input")

    def __init__(self, fn: str, input: Expr) -> None:
        self.fn = fn
        self.input = input

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.MATH, "fn": self.fn, "input": self.input.to_ir()}


class Math2Expr(Expr):
    """A two-argument math function (pow/atan2/round-to-digits) → Float64."""

    __slots__ = ("fn", "left", "right")

    def __init__(self, fn: str, left: Expr, right: Expr) -> None:
        self.fn = fn
        self.left = left
        self.right = right

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.MATH2,
            "fn": self.fn,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
        }


class Coalesce(Expr):
    """First non-null among the sub-expressions (SQL COALESCE)."""

    __slots__ = ("inputs",)

    def __init__(self, inputs: list[Expr]) -> None:
        self.inputs = inputs

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.COALESCE, "inputs": [e.to_ir() for e in self.inputs]}
