"""In-memory constructors: Python and Arrow objects to a lazy `Dataset`.

The column-oriented (`from_pydict`), row-oriented (`from_pylist`, `from_records`),
item-oriented (`from_items`), and streaming (`from_batches`, `from_iter`) entry
points, plus the Arrow and NumPy bridges. The names mirror Polars and pandas so a
ported script keeps its spelling: `from_dict`, `from_dicts`, and `from_records`
are the ecosystem-standard aliases.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.api.session._scan import _empty_batch, _scan
from batcher.io import interop
from batcher.io.source import InMemorySource, IteratorSource

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


def from_arrow(data: pa.Table | pa.RecordBatch | Sequence[pa.RecordBatch]) -> Dataset:
    """Create a `Dataset` from an Arrow table, record batch, or list of batches.

    Any object implementing the Arrow PyCapsule stream interface
    (``__arrow_c_stream__``) is accepted too, so a Polars frame, a DuckDB relation,
    or another engine's table can be handed over without naming its library.

    An empty (zero-row) table or batch is allowed — its schema is preserved via a
    single empty morsel, so an empty input flows through the engine like any other.
    A bare empty sequence of batches carries no schema and is rejected.

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
        PlanError: If `data` is an empty sequence carrying no schema.
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
                "sequence carries no schema; pass an empty pa.Table instead)"
            )
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


def from_pydict(mapping: Mapping[str, Any], *, schema: pa.Schema | None = None) -> Dataset:
    """Create a `Dataset` from a column-oriented ``{name: values}`` dict.

    Each key is a column and each value its list of cells (all the same length);
    types are inferred by Arrow unless `schema` pins them. The most direct way to
    get small in-memory data into the engine. Returns a lazy `Dataset` — no work
    runs until a terminal op.

    Args:
        mapping: Column name to its list of values (or NumPy array / Arrow array).
        schema: Declare the column types instead of inferring them.

    Returns:
        A lazy `Dataset` over the data.

    Raises:
        PlanError: If `mapping` is not a mapping, or its columns cannot be converted.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"region": ["w", "e"], "amount": [10, 20]})
            >>> ds.to_pydict()
            {'region': ['w', 'e'], 'amount': [10, 20]}
    """
    if not isinstance(mapping, Mapping):
        raise PlanError(
            f"from_pydict() expects a {{column: values}} mapping, got {type(mapping).__name__}; "
            "for a list of row dicts use bt.from_pylist()"
        )
    columns = dict(mapping)
    try:
        table = pa.table(columns, schema=schema)
    except _ARROW_CONVERSION_ERRORS as exc:
        table = _retry_as_tensors(columns, schema)
        if table is None:
            raise PlanError(_column_error("from_pydict", columns, exc)) from None
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


def _column_error(caller: str, columns: dict[str, Any], cause: Exception) -> str:
    """A message naming the column Arrow could not type, and the fix for what it holds.

    pyarrow quotes the offending value and its class and stops there, so a UUID primary key
    or an enum member in a fifty-column dict produced an error that named neither the column
    nor the remedy. The diagnosis is shared with the `map_batches` result path
    (`interop.diagnostics`): the same value is just as unconvertible on the way out.
    """
    from batcher.interop.diagnostics import describe_unconvertible, find_unconvertible_column

    name = find_unconvertible_column(columns)
    if name is None:
        return f"{caller}(): could not build an Arrow table — {cause}"
    return f"{caller}(): {describe_unconvertible(name, columns[name])}"


def from_pylist(rows: Sequence[Mapping[str, Any]]) -> Dataset:
    """Create a `Dataset` from a row-oriented list of ``{column: value}`` dicts.

    The row-major counterpart to `from_pydict` (e.g. JSON records); the union of keys
    is the schema and missing keys are null. ``bt.from_pylist([{"a": 1}, {"a": 2}])``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).to_pydict()
            {'a': [1, 2], 'b': ['x', 'y']}

    Args:
        rows: A list of ``{column: value}`` dicts; the union of keys is the schema.

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
    listed = list(rows)
    try:
        return _scan(interop.from_pylist(listed))
    except _ARROW_CONVERSION_ERRORS as exc:
        raise PlanError(_column_error("from_pylist", _as_columns(listed), exc)) from None


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
) -> Dataset:
    """Create a `Dataset` from a list of row tuples or row dicts (pandas spelling).

    Mirrors ``pd.DataFrame.from_records`` / ``pl.from_records``: tuple or list rows
    need `columns` to name them, dict rows do not. The common shape returned by a
    DB-API ``cursor.fetchall()``.

    Args:
        rows: The rows, each a tuple/list of values or a ``{column: value}`` dict.
        columns: Column names for tuple rows; required unless the rows are dicts.

    Returns:
        A lazy `Dataset` over the rows.

    Raises:
        PlanError: If tuple rows are given without `columns`, or a row's width does
            not match `columns`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_records([(1, "a"), (2, "b")], columns=["n", "s"]).to_pydict()
            {'n': [1, 2], 's': ['a', 'b']}
    """
    rows = list(rows)
    if rows and isinstance(rows[0], Mapping):
        return from_pylist(rows)
    if columns is None:
        raise PlanError(
            "from_records(): tuple rows carry no column names — pass columns=[...] "
            "(or use bt.from_pylist() for dict rows)"
        )
    names = list(columns)
    bad = next((r for r in rows if len(r) != len(names)), None)
    if bad is not None:
        raise PlanError(
            f"from_records(): row has {len(bad)} value(s) but {len(names)} column name(s) "
            f"were given ({names})"
        )
    return from_pydict({name: [row[i] for row in rows] for i, name in enumerate(names)})


def from_items(items: Sequence[Any], *, column: str = "item") -> Dataset:
    """Create a `Dataset` from a list of items, one row per item (Ray Data style).

    Dict items expand to columns (like `from_pylist`); scalar/other items become a
    single `column`. ``bt.from_items([1, 2, 3])`` / ``bt.from_items([{"a": 1}])``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_items([1, 2, 3]).to_pydict()
            {'item': [1, 2, 3]}

    Args:
        items: The items, one row each; dict items expand to columns.
        column: The single-column name used for scalar (non-dict) items.

    Returns:
        A lazy `Dataset` with one row per item.

    Raises:
        PlanError: If the items cannot become one Arrow column — most often because they
            are row tuples or Arrow batches, each of which has its own constructor.
    """
    rows = list(items)
    try:
        return _scan(interop.from_items(rows, column=column))
    except _ARROW_CONVERSION_ERRORS as exc:
        raise PlanError(_items_error("from_items", rows, exc)) from None


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
        "row tuples carry no column names — use bt.from_records(rows, columns=[...])",
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
) -> Dataset:
    """Create a `Dataset` from any Python iterable or generator, one row per item.

    The lazy-scripting entry point: a generator, a ``map``/``filter`` object, or a
    range is drained once into Arrow. Dict items expand to columns, scalars become a
    single `column`. Pass a zero-argument *callable* returning a fresh iterator when
    the source must be re-readable; for Arrow batches use `from_batches`, which
    streams in bounded memory instead of materializing.

    Args:
        iterable: The items, or a callable returning a fresh iterator of them.
        column: The single-column name used for scalar (non-dict) items.

    Returns:
        A lazy `Dataset` with one row per item.

    Raises:
        PlanError: If `iterable` is not iterable and not callable.

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
    rows = list(iterable)
    try:
        return _scan(interop.from_items(rows, column=column))
    except _ARROW_CONVERSION_ERRORS as exc:
        raise PlanError(_items_error("from_iter", rows, exc)) from None


def from_batches(
    factory: Callable[[], Iterator[pa.RecordBatch]],
    schema: pa.Schema | None = None,
    *,
    bounded: bool = True,
) -> Dataset:
    """Create a streaming `Dataset` from a re-iterable batch factory.

    `factory()` must return a fresh iterator of `pyarrow.RecordBatch` each call.
    Combined with `Dataset.iter_batches()`, a breaker-free pipeline (filter /
    project / map_batches) over this source is consumed one batch at a time in
    bounded memory — the path for unbounded or larger-than-memory inputs.

    A plain list of batches is accepted too, and `schema` may be omitted when the
    first batch can be drawn to read it (only for a bounded, re-iterable factory).
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

    Args:
        factory: A callable returning a fresh iterator of record batches each call,
            or a concrete sequence of record batches.
        schema: The Arrow schema of the produced batches; inferred from the first
            batch when omitted.
        bounded: Whether the stream is finite; ``False`` makes materializing terminal
            operations fail fast rather than hang.

    Returns:
        A streaming lazy `Dataset` over the factory's batches.

    Raises:
        PlanError: If `schema` is omitted and cannot be inferred.
    """
    if not callable(factory):
        return from_arrow(list(factory))
    if schema is None:
        first = next(iter(factory()), None)
        if first is None:
            raise PlanError(
                "from_batches(): cannot infer a schema from an empty factory — pass schema="
            )
        schema = first.schema
    return _scan(IteratorSource(factory, schema, bounded=bounded))


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
