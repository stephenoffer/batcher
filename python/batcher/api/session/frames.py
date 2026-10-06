"""In-memory constructors: Python and Arrow objects to a lazy `Dataset`.

The column-oriented (`from_pydict`), row-oriented (`from_pylist`, `from_records`),
item-oriented (`from_items`), and streaming (`from_batches`, `from_iter`) entry
points, plus the Arrow and NumPy bridges. The names mirror Polars and pandas so a
ported script keeps its spelling: `from_dict`, `from_dicts`, and `from_records`
are the ecosystem-standard aliases.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.api.session._scan import _empty_batch, _scan
from batcher.interop.diagnostics import (
    describe_unconvertible,
    find_unconvertible_column,
    is_scalar_column,
)
from batcher.io import interop
from batcher.io.source import InMemorySource, IteratorSource
from batcher.plan.types.registry import resolve_dtype_spec

__all__ = [
    "from_arrow",
    "from_batches",
    "from_items",
    "from_iter",
    "from_numpy",
    "from_pydict",
    "from_pylist",
    "from_records",
]


def _as_schema(caller: str, schema: pa.Schema | Mapping[str, Any] | None) -> pa.Schema | None:
    """`schema` as a pyarrow schema, read with the one dtype parser ``cast`` uses.

    What every in-memory constructor's ``schema=`` takes: a pyarrow schema, or a
    ``{column: dtype}`` mapping whose dtypes are spelled exactly as ``Expr.cast`` spells
    them (``"int32"``, ``"decimal(10,2)"``, ``int``) or are pyarrow types (needed for the
    nested ones the cast grammar cannot spell).
    """
    if schema is None or isinstance(schema, pa.Schema):
        return schema
    if not isinstance(schema, Mapping):
        raise PlanError(
            f"{caller}(): schema must be a pyarrow.Schema or a {{column: dtype}} mapping, "
            f"got {type(schema).__name__}"
        )
    fields = []
    for name, spec in schema.items():
        dtype = resolve_dtype_spec(spec, caller=caller)
        if dtype is None:
            raise PlanError(
                f"{caller}(): schema column {name!r} names unknown dtype {spec!r}; use a cast "
                "name such as 'int64' or 'decimal(10,2)', or a pyarrow type such as "
                "pa.list_(pa.int64())"
            )
        fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def _reject_duplicate_names(caller: str, names: Sequence[str]) -> None:
    """Refuse a relation whose column names repeat: one of them would silently vanish."""
    seen: set[str] = set()
    dupes = sorted({n for n in names if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
    if dupes:
        raise PlanError(
            f"{caller}(): column name(s) {dupes} appear more than once; every column of a "
            "Dataset needs a distinct name, so rename them first"
        )


def from_arrow(data: pa.Table | pa.RecordBatch | Sequence[pa.RecordBatch]) -> Dataset:
    """Create a `Dataset` from an Arrow table, record batch, or list of batches.

    Any object implementing the Arrow PyCapsule stream interface
    (``__arrow_c_stream__``) is accepted too, so a Polars frame, a DuckDB relation,
    or another engine's table can be handed over without naming its library.

    An empty (zero-row) table or batch is allowed — its schema is preserved via a
    single empty morsel, so an empty input flows through the engine like any other.
    A bare empty sequence of batches carries no schema and is rejected; pass
    ``schema.empty_table()`` instead.

    The data is held at construction: a ``__arrow_c_stream__`` producer is drained into
    a table there and then. To stream a producer in bounded memory instead, pass it to
    `from_batches`, which reads it at execution.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> import batcher as bt
            >>> bt.from_arrow(pa.table({"x": [1, 2]})).to_pydict()
            {'x': [1, 2]}

    Args:
        data: An Arrow table, record batch, sequence of record batches, or any
            object exporting ``__arrow_c_stream__``.

    Returns:
        A lazy `Dataset` over the Arrow data.

    Raises:
        PlanError: If `data` is an empty sequence carrying no schema, or two of its
            columns share a name.
    """
    if not isinstance(data, (pa.Table, pa.RecordBatch)) and hasattr(data, "__arrow_c_stream__"):
        data = pa.table(data)
    if isinstance(data, pa.Table):
        # A zero-row Table yields no batches; keep its schema with one empty morsel.
        batches = data.to_batches() or [_empty_batch(data.schema)]
    elif isinstance(data, pa.RecordBatch):
        batches = [data]
    else:
        batches = list(data)
        if not batches:
            raise PlanError(
                "from_arrow() requires at least one record batch (a bare empty "
                "sequence carries no schema; pass schema.empty_table() instead)"
            )
    _reject_duplicate_names("from_arrow", batches[0].schema.names)
    return _scan(InMemorySource(batches))


#: What a failed Arrow conversion raises, and the reason `ArrowNotImplementedError` belongs
#: in it. The three types this started as are what pyarrow raises for a value it can *type*
#: but not *hold*; a dtype it has no column form for at all — a NumPy structured array, the
#: `void` kind — raises `ArrowNotImplementedError`, which derives from `NotImplementedError`
#: rather than from `ValueError`. So it escaped every one of these handlers, and the column
#: diagnosis, the tensor retry, and the `PlanError` wrapping were all skipped for the case
#: that most needed them: the caller got a bare ``Unsupported numpy type 20`` naming neither
#: the column nor the constructor it came from.
_ARROW_CONVERSION_ERRORS = (
    pa.ArrowInvalid,
    pa.ArrowTypeError,
    pa.ArrowNotImplementedError,
    TypeError,
)


def from_pydict(
    mapping: Mapping[str, Any], *, schema: pa.Schema | Mapping[str, Any] | None = None
) -> Dataset:
    """Create a `Dataset` from a column-oriented ``{name: values}`` dict.

    Each key is a column and each value its list of cells (all the same length);
    types are inferred by Arrow unless `schema` pins them. The most direct way to
    get small in-memory data into the engine. The values are converted to Arrow and
    held at construction; no query work runs until a terminal op.

    `schema` is a pyarrow schema or a ``{column: dtype}`` mapping, spelled as
    ``Expr.cast`` spells a dtype or as a pyarrow type. It decides the columns and their
    order: a key the schema does not name is left out. An empty mapping with a schema
    is the typed empty `Dataset`, nested types included. Every in-memory constructor
    takes the same `schema`; for a strict missing/extra-column policy, use
    `Dataset.match_to_schema` on the result.

    Args:
        mapping: Column name to its list of values (or NumPy array / Arrow array).
        schema: Declare the column types (and order) instead of inferring them.

    Returns:
        A lazy `Dataset` over the data.

    Raises:
        PlanError: If `mapping` is not a mapping, a value is a single value rather than a
            column, a schema column is missing from `mapping`, or a column cannot be
            converted.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"region": ["w", "e"], "amount": [10, 20]})
            >>> ds.to_pydict()
            {'region': ['w', 'e'], 'amount': [10, 20]}

            >>> empty = bt.from_pydict({}, schema={"id": "int64", "name": "string"})
            >>> empty.count(), [str(t) for t in empty.dtypes]
            (0, ['int64', 'string'])
    """
    if not isinstance(mapping, Mapping):
        raise PlanError(
            f"from_pydict() expects a {{column: values}} mapping, got {type(mapping).__name__}; "
            "for a list of row dicts use bt.from_pylist()"
        )
    declared = _as_schema("from_pydict", schema)
    columns = dict(mapping)
    scalar = next((n for n, v in columns.items() if is_scalar_column(v)), None)
    if scalar is not None:
        raise PlanError(f"from_pydict(): {describe_unconvertible(scalar, columns[scalar])}")
    if declared is not None:
        if not columns:
            return from_arrow(declared.empty_table())
        missing = [n for n in declared.names if n not in columns]
        if missing:
            raise PlanError(
                f"from_pydict(): schema column(s) {missing} are not in the mapping; every "
                "schema column needs its values (pass {} for a typed empty Dataset)"
            )
    try:
        table = pa.table(columns, schema=declared)
    except _ARROW_CONVERSION_ERRORS as exc:
        table = _retry_as_tensors(columns, declared)
        if table is None:
            raise PlanError(_column_error("from_pydict", columns, exc, declared)) from None
    return from_arrow(table)


def _retry_as_tensors(columns: dict[str, Any], schema: pa.Schema | None) -> pa.Table | None:
    """Rebuild `columns`, turning a column of NumPy arrays into the tensor column that fits.

    Attempted only after a plain conversion has already failed, so the happy path pays
    nothing: a column of numbers never reaches here. A list of same-shape arrays becomes the
    canonical fixed-shape tensor column; a list of mixed-shape ones — the mixed-resolution
    image decode — becomes a variable-shape tensor column. Both used to be answered with
    "convert it to an ndarray", which the caller had already done.
    """
    from batcher.io.formats.ml.ragged import ragged_from_values
    from batcher.io.formats.ml.tensor import tensor_from_values

    converted = {
        name: _column_from_ndarray(value) or tensor_from_values(value) or ragged_from_values(value)
        for name, value in columns.items()
    }
    if not any(v is not None for v in converted.values()):
        return None
    rebuilt = {name: converted[name] or value for name, value in columns.items()}
    try:
        return pa.table(rebuilt, schema=schema)
    except _ARROW_CONVERSION_ERRORS:
        return None


def _column_from_ndarray(value: Any) -> pa.Array | None:
    """A multi-dimensional NumPy column converted by the `from_numpy` rank rules, else `None`.

    ``{"emb": np.random.rand(n, 384)}`` is how an embedding table is built from NumPy, and it
    is the one spelling of it that failed: `tensor_from_values` takes a *sequence* of per-row
    arrays and an `ndarray` is not a `collections.abc.Sequence`, so a bare N-D array declined
    and the caller was told the column "holds a sequence of numpy.ndarray, which Arrow cannot
    represent. Convert it to ... an ndarray" — advice they had already followed.

    Routing it through `io.interop.numpy_to_column` is what makes the two doors agree:
    ``bt.from_numpy(a)`` and ``bt.from_pydict({"x": a})`` now give the same array the same
    column type, where the first worked and the second raised.

    A **structured** array is claimed at any rank, because a compound dtype is a struct
    column and never converts on the first attempt whatever its shape. As a whole `Dataset`
    it is a table (`bt.from_numpy`); named as one column among others it is that column.

    Plain 1-D arrays are left alone deliberately. They convert on the first attempt and never
    reach here, and claiming them would put an untested second path under every ordinary
    numeric column for no gain.
    """
    import numpy as np

    if not isinstance(value, np.ndarray):
        return None
    if value.ndim < 2 and value.dtype.names is None:
        return None
    from batcher.io.interop import numpy_to_column

    return numpy_to_column(value)


def _column_error(
    caller: str, columns: dict[str, Any], cause: Exception, schema: pa.Schema | None = None
) -> str:
    """A message naming the column Arrow could not type, and the fix for what it holds.

    pyarrow quotes the offending value and its class and stops there, so a UUID primary key
    or an enum member in a fifty-column dict produced an error that named neither the column
    nor the remedy. The diagnosis is shared with the `map_batches` result path
    (`interop.diagnostics`): the same value is just as unconvertible on the way out. Under a
    declared `schema`, only its columns are examined and each against its declared type,
    since a key the schema leaves out never reached Arrow.
    """
    if schema is not None:
        columns = {n: columns[n] for n in schema.names if n in columns}
    name = find_unconvertible_column(columns, schema)
    if name is None:
        return f"{caller}(): could not build an Arrow table — {cause}"
    declared = schema.field(name).type if schema is not None else None
    return f"{caller}(): {describe_unconvertible(name, columns[name], declared)}"


def from_pylist(
    rows: Sequence[Mapping[str, Any]], *, schema: pa.Schema | Mapping[str, Any] | None = None
) -> Dataset:
    """Create a `Dataset` from a row-oriented list of ``{column: value}`` dicts.

    The row-major counterpart to `from_pydict` (e.g. JSON records); the union of keys
    is the schema and missing keys are null. The rows are converted to Arrow and held
    at construction, so a generator passed here is drained immediately.

    With `schema`, its columns are the result's, in its order: a missing key is null
    and a key the schema does not name is dropped. That is the lenient form; for a
    strict one, build without `schema` and call
    ``.match_to_schema(schema, extra_columns="raise")``. An empty list with a schema
    is the typed empty `Dataset`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).to_pydict()
            {'a': [1, 2], 'b': ['x', 'y']}

            >>> [str(t) for t in bt.from_pylist([], schema={"a": "float64"}).dtypes]
            ['double']

    Args:
        rows: A list of ``{column: value}`` dicts; the union of keys is the schema.
        schema: Declare the columns and their types instead of inferring them.

    Returns:
        A lazy `Dataset` over the rows.

    Raises:
        PlanError: If `rows` is a mapping (the column-oriented shape) rather than a
            sequence of row dicts, or if a column holds values Arrow cannot type.
    """
    if isinstance(rows, Mapping):
        raise PlanError(
            "from_pylist() expects a list of row dicts, got a mapping; "
            "for {column: values} use bt.from_pydict()"
        )
    declared = _as_schema("from_pylist", schema)
    listed = list(rows)
    try:
        return _scan(interop.from_pylist(listed, schema=declared))
    except _ARROW_CONVERSION_ERRORS as exc:
        raise PlanError(_column_error("from_pylist", _as_columns(listed), exc, declared)) from None


def _as_columns(rows: list) -> dict[str, list]:
    """Row dicts pivoted to ``{column: values}``, so the column diagnosis has columns to look at.

    Only ever built on the error path: the rows have already failed to convert, and finding
    *which* column did it is worth one pass over data that is not going anywhere.
    """
    names: dict[str, None] = {}
    for row in rows:
        if isinstance(row, Mapping):
            names.update(dict.fromkeys(row))
    return {name: [row.get(name) for row in rows if isinstance(row, Mapping)] for name in names}


def from_records(
    rows: Sequence[Any],
    *,
    columns: Sequence[str] | None = None,
    schema: pa.Schema | Mapping[str, Any] | None = None,
) -> Dataset:
    """Create a `Dataset` from a list of row tuples, row dicts, or dataclass instances.

    Mirrors ``pd.DataFrame.from_records`` / ``pl.from_records``. The common shape
    returned by a DB-API ``cursor.fetchall()``. Column names come from `columns` when
    given; otherwise from a namedtuple's ``_fields``, a dict's keys, a dataclass's
    fields, or `schema`'s names, and plain tuple rows with none of those are refused.
    The rows are converted and held at construction.

    A dataclass instance is read with ``dataclasses.asdict``, so the conversion is
    field by field: a nested dataclass becomes a struct column, a list a list column,
    an ``Optional`` field that is ``None`` a null, and a ``datetime``/``date`` a
    timestamp/date column. Arrow has no enum type, so an ``Enum`` field is refused
    with a message naming the fix (store its ``.value``). Types are inferred from the
    values, not from the annotations; pass `schema` to pin them, which also gives an
    all-``None`` column a type.

    Args:
        rows: The rows: tuples/lists of values, ``{column: value}`` dicts, or
            dataclass instances.
        columns: Column names for tuple rows, in order; each must be distinct.
        schema: Declare the column types (and, for tuple rows, the names) instead of
            inferring them.

    Returns:
        A lazy `Dataset` over the rows.

    Raises:
        PlanError: If tuple rows carry no names, `columns` repeats a name, a row's
            width does not match the names, or the rows mix dataclass and other kinds.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_records([(1, "a"), (2, "b")], columns=["n", "s"]).to_pydict()
            {'n': [1, 2], 's': ['a', 'b']}

            >>> from collections import namedtuple
            >>> Point = namedtuple("Point", ["x", "y"])
            >>> bt.from_records([Point(1, 2), Point(3, 4)]).to_pydict()
            {'x': [1, 3], 'y': [2, 4]}

            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Reading:
            ...     sensor: str
            ...     value: float | None
            >>> bt.from_records([Reading("a", 1.5), Reading("b", None)]).to_pydict()
            {'sensor': ['a', 'b'], 'value': [1.5, None]}
    """
    rows = list(rows)
    declared = _as_schema("from_records", schema)
    first = rows[0] if rows else None
    if _is_dataclass_row(first):
        return from_pylist(_dataclass_rows(rows), schema=declared)
    if isinstance(first, Mapping):
        return from_pylist(rows, schema=declared)
    names = _record_names(first, columns, declared)
    _reject_duplicate_names("from_records", names)
    bad = next((r for r in rows if len(r) != len(names)), None)
    if bad is not None:
        raise PlanError(
            f"from_records(): row has {len(bad)} value(s) but {len(names)} column name(s) "
            f"were given ({names})"
        )
    return from_pydict(
        {name: [row[i] for row in rows] for i, name in enumerate(names)}, schema=declared
    )


def _is_dataclass_row(row: object) -> bool:
    """Whether `row` is a dataclass *instance* (a dataclass class is not a row)."""
    return dataclasses.is_dataclass(row) and not isinstance(row, type)


def _dataclass_rows(rows: list) -> list[dict[str, Any]]:
    """Dataclass rows as dicts, refusing a list that mixes them with other kinds of row."""
    other = next((r for r in rows if not _is_dataclass_row(r)), None)
    if other is not None:
        raise PlanError(
            f"from_records(): the rows mix dataclass instances with {type(other).__name__}; "
            "pass rows of one kind"
        )
    return [dataclasses.asdict(r) for r in rows]


def _record_names(first: object, columns: Sequence[str] | None, schema: pa.Schema | None) -> list:
    """The column names for tuple rows: explicit, then a namedtuple's, then the schema's."""
    if columns is not None:
        return list(columns)
    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return list(first._fields)
    if schema is not None:
        return list(schema.names)
    raise PlanError(
        "from_records(): tuple rows carry no column names — pass columns=[...] "
        "(or use bt.from_pylist() for dict rows)"
    )


def from_items(
    items: Sequence[Any],
    *,
    column: str = "item",
    schema: pa.Schema | Mapping[str, Any] | None = None,
) -> Dataset:
    """Create a `Dataset` from a list of items, one row per item (Ray Data style).

    Dict items expand to columns (like `from_pylist`); scalar/other items become a
    single `column`. ``bt.from_items([1, 2, 3])`` / ``bt.from_items([{"a": 1}])``.
    The items are converted and held at construction. With `schema`, dict items
    follow `from_pylist`'s rules, scalar items fill the schema's `column` field, and
    an empty list is the typed empty `Dataset`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_items([1, 2, 3]).to_pydict()
            {'item': [1, 2, 3]}

            >>> [str(t) for t in bt.from_items([1, 2], schema={"item": "float64"}).dtypes]
            ['double']

    Args:
        items: The items, one row each; dict items expand to columns.
        column: The single-column name used for scalar (non-dict) items.
        schema: Declare the column types instead of inferring them.

    Returns:
        A lazy `Dataset` with one row per item.

    Raises:
        PlanError: If the items cannot become one Arrow column — most often because they
            are row tuples or Arrow batches, each of which has its own constructor.
    """
    return _items_dataset("from_items", list(items), column, schema)


def _items_dataset(
    caller: str, rows: list, column: str, schema: pa.Schema | Mapping[str, Any] | None
) -> Dataset:
    """The `Dataset` over `rows`, one per item — the body `from_items` and `from_iter` share."""
    declared = _as_schema(caller, schema)
    scalar_items = bool(rows) and not all(isinstance(r, dict) for r in rows)
    if declared is not None and scalar_items and column not in declared.names:
        raise PlanError(
            f"{caller}(): the items are not dicts, so they fill the single column {column!r}, "
            f"but the schema names {declared.names}; name {column!r} in the schema or pass "
            "column=..."
        )
    try:
        return _scan(interop.from_items(rows, column=column, schema=declared))
    except _ARROW_CONVERSION_ERRORS as exc:
        if declared is not None and scalar_items:
            raise PlanError(_column_error(caller, {column: rows}, exc, declared)) from None
        raise PlanError(_items_error(caller, rows, exc)) from None


#: The item shapes that fail to become one column *and* have a better constructor waiting.
#: Each entry is ``(predicate, remedy)``. Without this, ``bt.from_items([(1, "a")])`` — the
#: `cursor.fetchall()` shape, and the most common thing to try — raised pyarrow's
#: ``Could not convert 'a' with type str: tried to convert to int64``, which names neither
#: the constructor, nor the item, nor the fact that a one-line fix exists.
_ITEM_REMEDIES = (
    (
        lambda item: isinstance(item, pa.RecordBatch | pa.Table),
        "these are Arrow batches, not rows — use bt.from_batches(lambda: iter(batches)), "
        "which streams them in bounded memory, or bt.from_arrow(table)",
    ),
    (
        lambda item: isinstance(item, tuple | list),
        "row tuples carry no column names — use bt.from_records(rows, columns=[...]) "
        "(a namedtuple's own field names are used when columns is omitted)",
    ),
    (
        lambda item: isinstance(item, Mapping),
        "the items are not all dicts, so they cannot share a schema — make every item a "
        "{column: value} dict, or pass only the scalar items",
    ),
)


def _items_error(caller: str, rows: list, cause: Exception) -> str:
    """A message naming the item shape and the constructor that takes it.

    Falls back to quoting pyarrow when the shape is not one of the known confusions, because
    an unrecognized shape still deserves the underlying reason rather than a shrug.
    """
    first = next(iter(rows), None)
    for matches, remedy in _ITEM_REMEDIES:
        if matches(first):
            return f"{caller}(): {remedy}."
    return (
        f"{caller}(): could not build a column from items of type {type(first).__name__} — {cause}"
    )


def from_iter(
    iterable: Iterable[Any] | Callable[[], Iterable[Any]],
    *,
    column: str = "item",
    schema: pa.Schema | Mapping[str, Any] | None = None,
) -> Dataset:
    """Create a `Dataset` from any Python iterable or generator, one row per item.

    A generator, a ``map``/``filter`` object, or a range is drained **once, at
    construction**, and the items are held in memory as Arrow; nothing is read
    lazily. Dict items expand to columns, scalars become a single `column`, and
    `schema` follows `from_items`'s rules. A zero-argument *callable* is called
    once, at construction, for the iterator. For data that must stream in bounded
    memory, produce Arrow batches and use `from_batches`, which reads at execution.

    Args:
        iterable: The items, or a callable returning an iterator of them.
        column: The single-column name used for scalar (non-dict) items.
        schema: Declare the column types instead of inferring them.

    Returns:
        A lazy `Dataset` with one row per item.

    Raises:
        PlanError: If `iterable` is not iterable and not callable, or its items cannot
            become Arrow columns.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_iter(x * x for x in range(4)).to_pydict()
            {'item': [0, 1, 4, 9]}
    """
    if callable(iterable):
        iterable = iterable()
    if isinstance(iterable, (str, bytes)) or not isinstance(iterable, Iterable):
        raise PlanError(
            f"from_iter() expects an iterable of rows, got {type(iterable).__name__}; "
            "wrap a single value in a list"
        )
    return _items_dataset("from_iter", list(iterable), column, schema)


def from_batches(
    factory: Callable[[], Iterator[pa.RecordBatch]] | Any,
    schema: pa.Schema | None = None,
    *,
    bounded: bool = True,
) -> Dataset:
    """Create a streaming `Dataset` from a re-iterable batch factory.

    `factory()` must return a fresh iterator of `pyarrow.RecordBatch` each call.
    Combined with `Dataset.iter_batches()`, a breaker-free pipeline (filter /
    project / map_batches) over this source is consumed one batch at a time in
    bounded memory — the path for unbounded or larger-than-memory inputs.

    When the input is read: `factory` is called once per execution, and once more at
    construction only when `schema` is omitted (to draw the first batch for it). A
    plain list of batches is accepted too, and held as it is.

    Any object exporting the Arrow PyCapsule stream interface (``__arrow_c_stream__``:
    a ``pyarrow.RecordBatchReader``, a Polars frame, a DuckDB relation) streams too:
    it is re-exported at each execution and read batch by batch, never collected into
    one table. A ``RecordBatchReader`` is single-shot, so a second execution over it
    raises; pass a factory returning a fresh reader to read it again.

    Pass ``bounded=False`` for a genuinely infinite stream so terminal operations
    that must materialize (`collect`, `count`, `to_*`) fail fast instead of hanging.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> import batcher as bt
            >>> schema = pa.schema([("x", pa.int64())])
            >>> ds = bt.from_batches(lambda: iter([pa.record_batch({"x": [1, 2, 3]})]), schema)
            >>> ds.count()
            3

            >>> reader = pa.RecordBatchReader.from_batches(
            ...     schema, [pa.record_batch({"x": [1, 2]})]
            ... )
            >>> bt.from_batches(reader).to_pydict()
            {'x': [1, 2]}

    Args:
        factory: A callable returning a fresh iterator of record batches each call,
            a concrete sequence of record batches, or an ``__arrow_c_stream__``
            producer.
        schema: The Arrow schema of the produced batches; inferred from the first
            batch (or the producer's stream) when omitted.
        bounded: Whether the stream is finite; ``False`` makes materializing terminal
            operations fail fast rather than hang.

    Returns:
        A streaming lazy `Dataset` over the factory's batches.

    Raises:
        PlanError: If `schema` is omitted and cannot be inferred, or the batches'
            column names repeat.
    """
    if hasattr(factory, "__arrow_c_stream__") and not callable(factory):
        stream = _ArrowStream(factory)
        schema = schema if schema is not None else stream.schema()
        _reject_duplicate_names("from_batches", schema.names)
        return _scan(IteratorSource(stream, schema, bounded=bounded))
    if not callable(factory):
        return from_arrow(list(factory))
    if schema is None:
        first = next(iter(factory()), None)
        if first is None:
            raise PlanError(
                "from_batches(): cannot infer a schema from an empty factory — pass schema="
            )
        schema = first.schema
    _reject_duplicate_names("from_batches", schema.names)
    return _scan(IteratorSource(factory, schema, bounded=bounded))


class _ArrowStream:
    """A batch factory over an ``__arrow_c_stream__`` producer, re-exported per execution.

    A table-like producer (a pyarrow ``Table``, a Polars frame) exports a fresh stream every
    time it is asked, so each execution reads it from the start. A ``RecordBatchReader`` *is*
    the stream: exporting it a second time hands over an exhausted reader, which reads as a
    valid empty result. That would be a silently wrong answer, so the second read of a
    reader raises instead.
    """

    __slots__ = ("_producer", "_read")

    def __init__(self, producer: Any) -> None:
        self._producer = producer
        self._read = False

    def schema(self) -> pa.Schema:
        """The producer's schema, read without drawing a batch."""
        if isinstance(self._producer, pa.RecordBatchReader):
            return self._producer.schema
        return pa.RecordBatchReader.from_stream(self._producer).schema

    def __call__(self) -> Iterator[pa.RecordBatch]:
        if isinstance(self._producer, pa.RecordBatchReader):
            if self._read:
                raise PlanError(
                    "from_batches(): this RecordBatchReader was already read by an earlier "
                    "execution and a reader is single-shot; pass a factory such as "
                    "lambda: make_reader() to read the data again"
                )
            self._read = True
            return iter(self._producer)
        return iter(pa.RecordBatchReader.from_stream(self._producer))


def from_numpy(ndarray: Any, *, column: str = "data") -> Dataset:
    """Create a single-column `Dataset` from a NumPy array under name `column`.

    The leading axis is the row axis: a 1-D array becomes a scalar column, an
    ``(n, dim)`` array a fixed-size-list column (the embedding convention), and a
    higher-rank array a fixed-shape-tensor column. Needs only ``numpy`` (core).

    Pass a ``{name: array}`` dict to build one column per array instead; each one follows
    the same rules, so ``{"id": ids, "emb": vectors}`` is an embedding table in one call.

    A **structured** array — one with a compound dtype, as ``np.genfromtxt``,
    ``np.rec.array`` and an h5py compound dataset produce — is NumPy's own table, so it
    becomes one column per field and `column` is unused. A **masked** array keeps its mask:
    a masked value becomes a null, not the fill sitting underneath it.

    Args:
        ndarray: The array to ingest; its first axis indexes rows. A mapping of
            name to array builds one column each, and a structured array one column
            per field.
        column: The name of the single output column.

    Returns:
        A lazy `Dataset` with one column over the array, or one per field.

    Raises:
        PlanError: If the array has no Arrow column form — a complex dtype, which Arrow
            does not represent, or a 0-d array, which has no row axis.

    Examples:
        .. doctest::

            >>> import numpy as np
            >>> import batcher as bt
            >>> bt.from_numpy(np.array([1, 2, 3])).to_pydict()
            {'data': [1, 2, 3]}

            >>> rows = np.array([(1, 2.5)], dtype=[("id", "i8"), ("score", "f8")])
            >>> bt.from_numpy(rows).to_pydict()
            {'id': [1], 'score': [2.5]}
    """
    if isinstance(ndarray, Mapping):
        return from_pydict(ndarray)
    return _scan(interop.from_numpy(ndarray, column=column))
