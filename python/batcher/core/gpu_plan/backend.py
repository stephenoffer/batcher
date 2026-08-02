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

__all__ = [
    "DfBackend",
    "Unsupported",
    "call_or_decline",
    "conform_empty_dtype",
    "sortable_key",
    "widen_narrow",
    "widened_type",
]


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


def call_or_decline(obj: Any, name: str, *args: Any, **kwargs: Any):
    """Invoke `obj.name(*args)`, declining rather than crashing when it is absent or refuses.

    cuDF's surface is a subset of pandas' — a reduction, an accessor method or a keyword pandas
    offers may simply not exist on the device. An `AttributeError` or a `NotImplementedError`
    from that escapes as a crash, and worse, is classified as a *backend defect* rather than as
    a decline (`api.terminal.gpu_backend.failure`), so an ordinary expression reports the GPU
    path as broken. Turning it into `Unsupported` routes the stage to the CPU engine, which is
    the contract every other case in this package follows.

    Args:
        obj: The object to call the method on.
        name: The method name.
        *args: Positional arguments for the call.
        **kwargs: Keyword arguments for the call.

    Returns:
        Whatever the method returns.

    Raises:
        Unsupported: When the method is absent, or refuses these arguments.
    """
    method = getattr(obj, name, None)
    if method is None:
        raise Unsupported(f"{type(obj).__name__}.{name}")
    try:
        return method(*args, **kwargs)
    except (TypeError, ValueError, NotImplementedError, AttributeError) as exc:
        raise Unsupported(f"{name}: {exc}") from exc


class DfBackend:
    """One dataframe library (cuDF or pandas) behind the small surface the translator needs.

    Constructed per execution. The only state it keeps is which columns arrived as Arrow DATEs,
    because neither library has a date type and both hand one back as a timestamp — see
    `remember_dates`.
    """

    __slots__ = ("_arrow_native", "_date_types", "lib")

    def __init__(self, lib: Any) -> None:
        """Wrap dataframe module `lib` (``cudf`` on a GPU, ``pandas`` for verification)."""
        self.lib = lib
        # cuDF reads Arrow natively and keeps the null mask; pandas needs the ArrowDtype
        # mapper below to do the same.
        self._arrow_native = hasattr(lib.DataFrame, "from_arrow")
        self._date_types: dict[str, Any] = {}

    def remember_dates(self, schema) -> None:
        """Record which of `schema`'s columns are Arrow DATEs, so `to_arrow` can restore them.

        Neither dataframe library has a calendar-day type: a `date32` becomes a datetime the
        moment it enters a frame, and comes back out of `to_arrow` as a `timestamp`. The values
        are right and the *column* is wrong, which is the failure this package is most careful
        about elsewhere — a shard that contributes a `timestamp` column cannot be concatenated
        with a CPU-recovered shard's `date32`, and a query that returns one has quietly changed
        its own schema. Measured on TPC-H q3, whose `o_orderdate` came back as
        `datetime(1995, 2, 3, 0, 0)` where the engine returns `date(1995, 2, 3)`.

        Called by every reader — `from_arrow` here, and the device Parquet reader, which never
        goes through it.

        Args:
            schema: The Arrow schema the rows were read with.
        """
        import pyarrow as pa

        for field in schema:
            if pa.types.is_date(field.type):
                self._date_types[field.name] = field.type

    def remember_date_alias(self, name: str, arrow_type: Any) -> None:
        """Record that the projection producing `name` is DATE-typed, so `to_arrow` restores it.

        `remember_dates` covers a date the plan *read*; this covers one it *computes* —
        `col("ts").dt.date()`, `last_day`, a day offset over a date. Neither library has a
        calendar-day type, so such a column comes back as a timestamp, and on the host backend
        the `astype` happens to land on `date32` anyway while on the device it cannot. The
        result was a column that was right in CI and a `timestamp[ms]` on a real GPU — the
        wrong-column failure a fan-out cannot concatenate, visible only on the device.

        Args:
            name: The output column's name.
            arrow_type: The Arrow DATE type it should be presented as.
        """
        self._date_types[name] = arrow_type

    def forget_date_alias(self, name: str) -> None:
        """Drop any DATE claim on `name`, because the projection now producing it is not one.

        A chain may reuse a name for a different expression, and the *last* projection decides
        what the column is. Without this, a date read early would keep casting a timestamp
        computed later under the same name.

        Args:
            name: The output column's name.
        """
        self._date_types.pop(name, None)

    def is_date_column(self, name: str) -> bool:
        """Whether `name` is known to be a DATE — asked when deciding what an expression over
        it produces.

        Args:
            name: The column's name.

        Returns:
            True when the column arrived as, or was computed as, an Arrow DATE.
        """
        return name in self._date_types

    @property
    def is_gpu(self) -> bool:
        """Whether this backend computes on a device (cuDF) rather than the host (pandas)."""
        return self._arrow_native

    def from_arrow(self, table: pa.Table):
        """An Arrow table as a dataframe, preserving Arrow's null mask on both libraries."""
        table = widen_narrow(table)
        self.remember_dates(table.schema)
        if self._arrow_native:
            return self.lib.DataFrame.from_arrow(table)
        return table.to_pandas(types_mapper=self.lib.ArrowDtype)

    def to_arrow(self, df) -> pa.Table:
        """A dataframe back as an Arrow table, dropping the index (never part of the result)."""
        if self._arrow_native:
            return self._restore_dates(self._restore_empty_strings(df, df.to_arrow()))
        import pyarrow as pa

        table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
        return self._restore_dates(table)

    def _restore_empty_strings(self, df, table: pa.Table) -> pa.Table:
        """An **empty** cuDF string column converts to Arrow `null`; give it back its type.

        Measured on the device: both `cudf.DataFrame({"s": ["a"]})[...filtered to empty...]` and
        an explicitly empty string column convert as `s: null`, while an `int64` column beside
        them keeps `int64`. So a query whose result is empty came back with every string column
        untyped, and TPC-H q15 — which is empty on this data — returned `s_name`, `s_address` and
        `s_phone` as `null` where the CPU engine returns `string`. Every value agreed, because
        there were none. That is the whole danger of it: the device tier's contract fixes column
        *types* as exactly as it fixes rows, and a value comparison can never see this.

        Safe because in cuDF `object` means `string` and nothing else — it has no arbitrary
        Python object column — so the dtype is a reliable statement of what the column held. The
        repair is confined to the empty case, where there is provably no data to misread, and to
        this backend, where that equivalence holds; a pandas frame's `object` column could be
        anything, and the host backend exists to mirror the engine's tests rather than to answer
        queries.
        """
        if table.num_rows:
            return table
        import pyarrow as pa

        columns = set(df.columns)
        fields, changed = [], False
        for field in table.schema:
            if (
                field.name in columns
                and pa.types.is_null(field.type)
                and str(df[field.name].dtype) == "object"
            ):
                fields.append(pa.field(field.name, pa.string(), nullable=True))
                changed = True
            else:
                fields.append(field)
        return table.cast(pa.schema(fields)) if changed else table

    def _restore_dates(self, table: pa.Table) -> pa.Table:
        """`table` with any column that arrived as a DATE cast back from the library's timestamp.

        Keyed by name and applied only where the column *is* now a timestamp, so a column the
        plan genuinely converted keeps its conversion and one that was never a date is untouched.
        A no-op — and not even a schema walk — when nothing this backend read was a date, which
        is most plans.
        """
        if not self._date_types:
            return table
        import pyarrow as pa

        fields = []
        changed = False
        for field in table.schema:
            target = self._date_types.get(field.name)
            if target is not None and pa.types.is_timestamp(field.type):
                fields.append(pa.field(field.name, target, field.nullable))
                changed = True
            else:
                fields.append(field)
        return table.cast(pa.schema(fields)) if changed else table

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

    def is_date(self, value: Any) -> bool:
        """Whether `value` is a DATE column — a calendar day with no time of day.

        Asked because subtracting two of them is the one arithmetic the engine answers in a
        different *unit* from the libraries: it returns a count of days, they return a duration.
        A timestamp difference is a duration on both, so only DATE needs the distinction.
        """
        if not self.is_series(value):
            return False
        dtype = getattr(value, "dtype", None)
        arrow = getattr(dtype, "pyarrow_dtype", None)
        if arrow is None:
            return False
        import pyarrow as pa

        return pa.types.is_date(arrow)

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

    def is_boolean(self, value: Any) -> bool:
        """Whether `value` is a boolean column or scalar.

        Asked because the bit operators answer in a different *type* over a boolean than the
        libraries do: the engine (and DuckDB) return an integer, both libraries route `&`, `|`
        and `^` to logical operators and return a boolean. Same values, wrong column.
        """
        if not self.is_series(value):
            return isinstance(value, bool)
        dtype = getattr(value, "dtype", None)
        arrow = getattr(dtype, "pyarrow_dtype", None)
        if arrow is not None:
            import pyarrow as pa

            return pa.types.is_boolean(arrow)
        return getattr(dtype, "kind", "") == "b"

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


# --- conforming a dataframe column to what the engine would return ------------------
#
# Both libraries disagree with the engine in ways that produce a right *value* in a wrong
# *column*, which is the failure this package is most exposed to: a sharded fan-out
# concatenates its shards' partials, and a shard that fell back to the CPU engine contributes
# the engine's type beside a device shard's. These two are the window operator's share of that,
# and `aggs._normalized_key` is the aggregate's — same fold, same reason.

#: Window functions whose result type is *not* their input's. Everything else — `sum`, `min`,
#: `max`, `product`, the bit/bool folds, and every value function — returns the input's own
#: type, which is what `_fix_empty_dtype` restores.
_WIDENS_TO_DOUBLE = frozenset({"avg", "var", "stddev", "percent_rank", "cume_dist"})
_COUNTS = frozenset({"count", "count_distinct", "row_number", "rank", "dense_rank", "ntile"})


def conform_empty_dtype(out, f: dict, name: str | None) -> None:
    """Give a window column computed over an *empty* frame the type a non-empty one would have.

    Both dataframe libraries type an empty reduction as `float64` regardless of what went in,
    so `SUM(v)` over an empty partition came back `double` where the engine returns `int64`.
    The values agree — there are none — and the *column* does not, which is the failure this
    package is most exposed to: a sharded fan-out cannot concatenate a device shard that
    contributed `double` with a CPU-recovered shard that contributed `int64`, and an empty
    shard is the single most likely one to fall back.

    Only the empty case is adjusted. A non-empty frame already carries the input's type
    through, and re-casting it would be a second, drifting statement of the engine's type
    rules rather than a repair of a library default.
    """
    if len(out) != 0:
        return
    alias, fn = f["alias"], f["func"]
    if fn in _WIDENS_TO_DOUBLE or fn in _COUNTS or name is None:
        return  # already double / already cast to int64 by the rank path
    out[alias] = out[alias].astype(out[name].dtype)


def sortable_key(out, name: str, be: DfBackend, computed: list[str], *, slot: int) -> str:
    """`name`, or a private copy of it safe to multi-key sort on.

    A float column holding both `-0.0` and `0.0` cannot go through pandas' multi-key sort:
    `lexsort_indexer` builds a `Categorical` per key and infers its categories from the unique
    values, and the two zeros are distinct bit patterns that compare equal — so the inferred
    categories are not unique and pandas raises a bare `ValueError`. That escaped the
    translator entirely, so an ordinary `PARTITION BY g ORDER BY k, f` dropped to the CPU
    engine, and did so as an unclassified exception rather than a decline.

    Folding the two zeros together is exactly what `aggs._normalized_key` does for a group
    key, and it is order-preserving for the same reason: IEEE says they are equal, so no sort
    can distinguish them anyway. Only float keys pay for the copy.
    """
    if not be.is_float(out[name]):
        return name
    normalized = f"__bt_wsk{slot}"
    out[normalized] = out[name] + 0.0
    computed.append(normalized)
    return normalized
