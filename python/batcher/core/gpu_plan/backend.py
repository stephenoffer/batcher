"""The dataframe-library adapter the GPU translator runs against.

Every kernel in this package is written once, against `DfBackend`, and runs on **cuDF** on a
GPU worker (the accelerated backend) or on **pandas** for the head-runnable correctness test
against the native CPU engine. That is the whole reason the GPU path is trustworthy: the GPU
is only *where* the translated plan runs, never *what* it computes, so a pandas replay in CI
proves the same code a GPU executes.

The adapter exists because the two libraries disagree on exactly the things a correctness
contract depends on. cuDF carries a real Arrow null mask; pandas' default `to_pandas()` melts
nulls into `NaN`, which makes `is_null` and `is_nan` indistinguishable and silently turns an
all-null `sum` into `0.0`. Loading through `pd.ArrowDtype` keeps pandas on Arrow semantics, so
the verification path models the device faithfully instead of approximately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["DfBackend", "Unsupported", "widen_narrow", "widened_type"]


def widened_type(dtype: pa.DataType):
    """The type the engine would present `dtype` as, or `None` when it presents it unchanged.

    The FFI boundary normalizes every integer width — signed and unsigned, `uint64` included —
    to `int64`, and every narrow float to `double`. A query over an `int32` column therefore
    returns `int64` from the engine, and every test and every downstream consumer is written
    against that.

    The translator reads Arrow *without* crossing that boundary, so left alone it hands back
    the source's own width. That is not a wrong number, but it is a wrong column, and it is
    worst exactly where this backend is used: a sharded fan-out concatenates its shards'
    partials, and a shard that fell back to the CPU engine contributes `int64` beside a device
    shard's `int32`.

    Args:
        dtype: The source column's Arrow type.

    Returns:
        The widened Arrow type, or `None` when `dtype` already is what the engine would show.
    """
    import pyarrow as pa

    if pa.types.is_integer(dtype) and dtype != pa.int64():
        return pa.int64()
    if pa.types.is_floating(dtype) and dtype != pa.float64():
        return pa.float64()
    return None


def widen_narrow(table: pa.Table) -> pa.Table:
    """`table` with its narrow numeric columns widened the way the engine's boundary would.

    A no-op — and not even a schema rebuild — for a table that is already wide, which is the
    common case, so the check costs one pass over the field list rather than over the data.

    Args:
        table: The table as read from storage.

    Returns:
        The table, cast where a column's width would otherwise disagree with the engine.
    """
    import pyarrow as pa

    targets = [widened_type(field.type) for field in table.schema]
    if not any(targets):
        return table
    fields = [
        field if target is None else pa.field(field.name, target, field.nullable)
        for field, target in zip(table.schema, targets, strict=True)
    ]
    return table.cast(pa.schema(fields))


class Unsupported(Exception):
    """An op or expression the translator does not handle — triggers the CPU fallback.

    Raised rather than approximated. Every caller turns it into "run this on the CPU engine",
    so an unsupported shape costs a fallback; a *silently approximated* one costs a wrong
    answer, which is the failure mode this whole package is built to avoid.
    """


class DfBackend:
    """One dataframe library (cuDF or pandas) behind the small surface the translator needs.

    Holds no state beyond the module, so it is free to construct per execution.
    """

    __slots__ = ("_arrow_native", "lib")

    def __init__(self, lib: Any) -> None:
        """Wrap dataframe module `lib` (``cudf`` on a GPU, ``pandas`` for verification)."""
        self.lib = lib
        # cuDF reads Arrow natively and keeps the null mask; pandas needs the ArrowDtype
        # mapper below to do the same.
        self._arrow_native = hasattr(lib.DataFrame, "from_arrow")

    @property
    def is_gpu(self) -> bool:
        """Whether this backend computes on a device (cuDF) rather than the host (pandas)."""
        return self._arrow_native

    def from_arrow(self, table: pa.Table):
        """An Arrow table as a dataframe, preserving Arrow's null mask on both libraries."""
        table = widen_narrow(table)
        if self._arrow_native:
            return self.lib.DataFrame.from_arrow(table)
        return table.to_pandas(types_mapper=self.lib.ArrowDtype)

    def to_arrow(self, df) -> pa.Table:
        """A dataframe back as an Arrow table, dropping the index (never part of the result)."""
        if self._arrow_native:
            return df.to_arrow()
        import pyarrow as pa

        return pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)

    def concat(self, frames: list):
        """Row-wise concatenation with a fresh index."""
        return self.lib.concat(frames, ignore_index=True)

    def series(self, values: list, dtype: Any = None):
        """A one-dimensional column from a Python list."""
        if dtype is not None:
            return self.lib.Series(values, dtype=dtype)
        return self.lib.Series(values)

    def float_series(self, raw):
        """A float column from a NumPy array, keeping `NaN` a value rather than a null.

        Both libraries' default constructors read a float `NaN` as *missing*, which erases
        exactly the distinction the engine cares about: `sqrt(NaN)` is `NaN`, not null.
        Going through Arrow (or cuDF's explicit `nan_as_null=False`) keeps the two apart.
        """
        if self._arrow_native:
            return self.lib.Series(raw, nan_as_null=False)
        import pyarrow as pa

        return self.lib.Series(self.lib.arrays.ArrowExtensionArray(pa.array(raw)))

    def dtype(self, arrow_type: pa.DataType):
        """`arrow_type` as the dtype this library's `astype` understands.

        pandas is given an `ArrowDtype` so a cast keeps Arrow's null mask — `astype("float64")`
        would drop to NumPy and turn every null into `NaN`, which `is_null` then reports as
        non-null. cuDF's dtypes are already Arrow-backed, so the NumPy-style name is exact.
        """
        if self._arrow_native:
            import pyarrow as pa

            if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                return "str"
            return arrow_type.to_pandas_dtype()
        return self.lib.ArrowDtype(arrow_type)

    def is_series(self, value: Any) -> bool:
        """Whether `value` is a column of this library rather than a Python scalar."""
        return isinstance(value, self.lib.Series)

    def is_float(self, value: Any) -> bool:
        """Whether `value` is a floating-point column or scalar — i.e. can hold `NaN`.

        The engine orders `NaN` as larger than every number and equal to itself (DuckDB's
        total order), while both dataframe libraries use IEEE comparison, where every `NaN`
        comparison is false. Reconciling that costs extra work, so it is done only where a
        `NaN` can actually occur.
        """
        if not self.is_series(value):
            return isinstance(value, float)
        dtype = getattr(value, "dtype", None)
        arrow = getattr(dtype, "pyarrow_dtype", None)
        if arrow is not None:
            import pyarrow as pa

            return pa.types.is_floating(arrow)
        return getattr(dtype, "kind", "") == "f"

    def is_integer(self, value: Any) -> bool:
        """Whether `value` is an integer column.

        Asked by exactly one caller, and for a narrow reason: `abs` is the only unary math
        function whose result keeps its input's integer type — every other one widens to
        double on both the engine and here. Routing an integer `abs` through the float ufunc
        path returns `1.0` where the engine returns `1`, which is not a wrong number but is a
        wrong *column*, and a shard that contributes one cannot be concatenated with its peers.
        """
        if not self.is_series(value):
            return isinstance(value, int) and not isinstance(value, bool)
        dtype = getattr(value, "dtype", None)
        arrow = getattr(dtype, "pyarrow_dtype", None)
        if arrow is not None:
            import pyarrow as pa

            return pa.types.is_integer(arrow)
        return getattr(dtype, "kind", "") in ("i", "u")

    def has_nan(self, value: Any) -> bool:
        """Whether a float column actually carries a `NaN` (as opposed to a null).

        `x != x` is true only for `NaN` and null only for null, so filling and reducing gives
        an exact answer. Used to *decline* the reductions whose `NaN` behavior the libraries
        disagree with the engine on, rather than to approximate them: the engine treats `NaN`
        as the largest value, and both libraries treat it as missing, so a `max` over a
        `NaN`-bearing group silently returns the largest *finite* value instead.
        """
        if not self.is_series(value) or not self.is_float(value):
            return False
        return bool((value != value).fillna(False).any())

    def broadcast(self, scalar: Any, df):
        """`scalar` as a full column aligned to `df`, materialized in the library's own layer.

        Never `[scalar] * len(df)`: that allocates one Python object per row, which
        `.claude/rules/architecture.md` forbids outright, and on cuDF it also forces a host-side
        list across the host/device boundary.

        The dtype must be inferred from a one-element *list* rather than `Series(scalar,
        index=...)`, which looks equivalent and is not: a `None` literal coerces to `float64`
        `NaN` instead of staying null, and a `datetime` lands at `datetime64[us]` instead of
        `datetime64[ns]` — either of which silently changes the column's type at the Arrow
        boundary.
        """
        col = self.lib.Series([scalar]).repeat(len(df))
        col.index = df.index
        return col

    def column(self, value: Any, df):
        """`value` as a column of `df`'s length, whether it arrived as a column or a scalar."""
        return value if self.is_series(value) else self.broadcast(value, df)
