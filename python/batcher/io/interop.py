"""Framework-interop ingestion — build a `Source` from a foreign object.

Ray-Data-style ``from_*`` constructors that adapt an in-memory object from a
neighboring framework (Arrow, pandas, Polars, NumPy, HuggingFace, PyTorch,
TensorFlow, Spark, Dask) into a Batcher `Source`. Every adapter normalizes to
Arrow and returns an `InMemorySource` (eager, materialized) or an
`IteratorSource` (lazy, streaming) — it never builds a `Dataset` (the session
layer wraps these Sources), so importing this module pulls in no optional
framework and creates no import cycle.

The conversion is **batch-granular and zero-copy where the framework allows it**
(HuggingFace datasets and Polars are Arrow-backed; pandas/Spark go through their
native Arrow bridges). Per-row Python is never used to move data.

``from_arrow`` / ``from_pydict`` / ``from_numpy`` are CORE (only pyarrow / numpy).
Every other adapter defers its import through `_internal.optional.require`, the engine's one
optional-dependency guard, so an absent framework raises `MissingDependencyError` naming the
framework and the exact ``pip install`` that fixes it.

Going through `require` rather than a local helper closes two gaps this module had. Its own
guard raised a plain `BackendError`, so ``except ImportError`` around ``bt.from_pandas(df)``
did not catch what the identical spelling around ``bt.read.parquet(...)`` does, and the error
carried no `install` field for a caller wanting to surface the command its own way. And two of
the hints named extras that **did not exist** — ``batcher-engine[spark]`` and ``[dask]`` — so
the one actionable thing in the message was a command that fails; both are now declared in
`pyproject.toml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher._internal.optional import require
from batcher.io.source import InMemorySource, IteratorSource, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "from_arrow",
    "from_dask",
    "from_huggingface",
    "from_items",
    "from_numpy",
    "from_pandas",
    "from_polars",
    "from_pydict",
    "from_pylist",
    "from_ray_dataset",
    "from_spark",
    "from_tf",
    "from_torch",
    "numpy_to_column",
    "structured_to_columns",
]


def _source_from_table(table: pa.Table) -> Source:
    """Wrap a (possibly empty) Arrow table as an `InMemorySource`.

    An empty table still has a schema, so it becomes a single empty batch — a
    source must always expose at least one batch to publish its schema.
    """
    batches = table.to_batches()
    if not batches:
        # A zero-row table yields no batches; build one empty batch that carries the
        # schema. It MUST have one (empty) array per field — `from_arrays([], schema)`
        # with a non-empty schema raises "Schema and number of arrays unequal".
        empty = [pa.array([], type=f.type) for f in table.schema]
        batches = [pa.RecordBatch.from_arrays(empty, schema=table.schema)]
    return InMemorySource(batches)


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a table from row dicts using the ORDERED UNION of keys as the schema.

    ``pa.Table.from_pylist`` infers the schema from the first row alone, so a key
    that appears only in a later row is silently dropped and its values lost. This
    takes every key across every row (first-seen order) and fills missing cells with
    null — the documented row-ingestion contract. It is column-oriented, so each
    column's type is inferred independently.
    """
    keys: dict[str, None] = {}
    for row in rows:
        for k in row:
            keys.setdefault(k)
    return pa.table({k: [row.get(k) for row in rows] for k in keys})


# ---- CORE adapters (no optional dependency) ------------------------------
def from_arrow(table_or_batches: pa.Table | pa.RecordBatch | list[pa.RecordBatch]) -> Source:
    """Build a `Source` from a pyarrow `Table`, `RecordBatch`, or batch list.

    Zero-copy: the Arrow data is referenced directly, never re-serialized.
    """
    if isinstance(table_or_batches, pa.Table):
        return _source_from_table(table_or_batches)
    if isinstance(table_or_batches, pa.RecordBatch):
        return InMemorySource([table_or_batches])
    batches = list(table_or_batches)
    if not batches:
        raise PlanError(
            "from_arrow() needs at least one record batch: a Dataset is a handle to a "
            "schema as well as to rows, and an empty list carries neither. Pass a "
            "pa.Table (which has a schema even when empty), or bt.from_pydict({...})."
        )
    return InMemorySource(batches)


def from_pydict(data: dict[str, Any]) -> Source:
    """Build a `Source` from a column-oriented ``{name: values}`` dict."""
    return _source_from_table(pa.table(data))


def from_pylist(rows: list[dict[str, Any]]) -> Source:
    """Build a `Source` from a row-oriented list of ``{column: value}`` dicts.

    The row-major counterpart to `from_pydict` — the natural shape for JSON records or
    API responses. Missing keys become nulls; the union of keys is the schema.
    """
    return _source_from_table(_table_from_rows(rows))


def from_items(items: list[Any], *, column: str = "item") -> Source:
    """Build a `Source` from a list of items, one row per item (the Ray Data shape).

    Dict items expand to columns (like `from_pylist`); scalar/other items become a
    single `column`. ``from_items([1, 2, 3])`` → one ``item`` column;
    ``from_items([{"a": 1}, {"a": 2}])`` → an ``a`` column.
    """
    rows = list(items)
    if rows and all(isinstance(r, dict) for r in rows):
        return _source_from_table(_table_from_rows(rows))
    return _source_from_table(pa.table({column: rows}))


def from_numpy(ndarray: Any, *, column: str = "data") -> Source:
    """Build a `Source` from a NumPy array — one column named `column`, or one per field.

    The leading axis is the row axis. A 1-D array becomes a scalar column; an
    ``(n, dim)`` array becomes a ``FixedSizeList<…, dim>`` column (the embedding
    convention); an ``(n, *shape)`` array with ``shape`` of rank >= 2 becomes a
    fixed-shape-tensor column that preserves the full per-row shape.

    A **structured** array (one with a compound dtype: ``np.genfromtxt``, ``np.recarray``,
    an h5py compound dataset) is NumPy's own table, so it becomes one column per field and
    `column` is unused — the same reading `pandas.DataFrame` gives it. Each field goes
    through the rank rules above, so a sub-array field keeps its per-row shape.
    """
    columns = structured_to_columns(ndarray)
    if columns is not None:
        return _source_from_table(pa.table(columns))
    return InMemorySource([pa.RecordBatch.from_arrays([numpy_to_column(ndarray)], names=[column])])


# ---- optional-framework adapters -----------------------------------------
def from_pandas(df: Any) -> Source:
    """Build a `Source` from a pandas `DataFrame` via ``pa.Table.from_pandas``.

    The pandas index is dropped (``preserve_index=False``) — matching DuckDB, Polars,
    and Ray Data. Keeping it would leak pyarrow's internal ``__index_level_0__``
    column (or the index name) into the public schema as a phantom extra column; call
    ``df.reset_index()`` first to ingest the index as a real column.
    """
    require("pandas", feature="pandas interop", provides="pandas", extra="pandas")
    return _source_from_table(pa.Table.from_pandas(df, preserve_index=False))


def from_polars(df: Any) -> Source:
    """Build a `Source` from a Polars `DataFrame` via its zero-copy Arrow export."""
    require("polars", feature="Polars interop", provides="polars", extra="polars")
    return _source_from_table(df.to_arrow())


def from_huggingface(hf_dataset: Any) -> Source:
    """Build a `Source` from a HuggingFace `datasets.Dataset` (Arrow-backed).

    HuggingFace datasets are Arrow tables under the hood, so the underlying table
    is taken directly (zero-copy) — falling back to ``with_format('arrow')`` for
    dataset views that do not expose ``.data`` directly.
    """
    require("datasets", feature="HuggingFace interop", provides="datasets", extra="huggingface")
    data = getattr(hf_dataset, "data", None)
    table = getattr(data, "table", None)
    if isinstance(table, pa.Table):
        return _source_from_table(table)
    arrow_ds = hf_dataset.with_format("arrow")
    return _source_from_table(pa.Table.from_batches(list(arrow_ds.iter(batch_size=1024))))


def from_torch(dataset_or_tensors: Any) -> Source:
    """Build a `Source` from a PyTorch tensor, a mapping/tuple of tensors, or a `Dataset`.

    Tensors are moved to CPU and adapted via NumPy (one column per tensor); an
    iterable `Dataset` of tensor rows is stacked column-wise. No per-row Python
    crosses into the engine — only the bulk NumPy buffers do.

    A ``{name: tensor}`` **mapping** keeps its keys as column names, which makes this the
    exact inverse of `ml.to_torch`: what the loader yields, this reads back. Before, a
    mapping fell through to the map-style-`Dataset` branch and indexed it by integer, so the
    natural round-trip raised ``KeyError: 0`` — an error naming nothing the caller wrote.

    A tuple or list of tensors becomes ``col_0``, ``col_1``, … and now goes through the same
    rank rules as every other door, so ``(features, labels)`` with an ``(n, dim)`` feature
    tensor — the canonical PyTorch pair — becomes a `FixedSizeList` column instead of raising
    ``only handle 1-dimensional arrays``.
    """
    torch = require("torch", feature="PyTorch interop", provides="torch", extra="torch")

    def _np(t: Any) -> Any:
        return _tensor_to_numpy(t, torch)

    if isinstance(dataset_or_tensors, torch.Tensor):
        return from_numpy(_np(dataset_or_tensors))
    if isinstance(dataset_or_tensors, Mapping):
        named = {
            str(name): numpy_to_column(_np(value) if isinstance(value, torch.Tensor) else value)
            for name, value in dataset_or_tensors.items()
        }
        return _source_from_table(pa.table(named))
    if isinstance(dataset_or_tensors, (tuple, list)) and all(
        isinstance(t, torch.Tensor) for t in dataset_or_tensors
    ):
        cols = {f"col_{i}": numpy_to_column(_np(t)) for i, t in enumerate(dataset_or_tensors)}
        return _source_from_table(pa.table(cols))
    columns = _stack_torch_dataset(dataset_or_tensors, _np)
    return from_pydict(columns)


def _tensor_to_numpy(tensor: Any, torch: Any) -> Any:
    """One detached CPU `torch.Tensor` as a NumPy array, widening what NumPy cannot hold.

    ``Tensor.numpy()`` raises ``TypeError: Got unsupported ScalarType BFloat16`` for exactly
    the dtypes modern ML runs in — ``bfloat16`` is what almost every LLM checkpoint and every
    mixed-precision training loop carries, and ``float8_e4m3fn``/``float8_e5m2`` are what
    quantized inference emits. NumPy has no such dtype and neither does Arrow, so the tensor
    is widened to the narrowest type that holds it exactly (``float32``) and the widening is
    announced, rather than the ingest failing on a message that names a torch-internal enum.

    ``float32`` is exact for all three: bfloat16 and both float8 formats have fewer mantissa
    and no more exponent bits, so no value moves. What changes is the column's width on the
    wire, which is why it is said out loud rather than done quietly.

    Args:
        tensor: The tensor to convert.
        torch: The imported ``torch`` module (the caller already holds it).

    Returns:
        The NumPy array. Complex tensors are left to `numpy_to_column` to decline, so the
        complex remedy is stated in one place.
    """
    detached = tensor.detach().cpu()
    try:
        return detached.numpy()
    except TypeError:
        import warnings

        name = str(detached.dtype).removeprefix("torch.")
        warnings.warn(
            f"a torch.{name} tensor has no NumPy or Arrow dtype, so it is widened to "
            "float32 on the way in. No value moves — float32 has more mantissa bits and no "
            f"fewer exponent bits than {name} — but the column is 4 bytes a value rather "
            "than 1 or 2. Cast it yourself first (`t.to(torch.float16)`) for a narrower "
            "column, or keep the tensor out of the engine if the width matters more than "
            "the query.",
            UserWarning,
            stacklevel=6,
        )
        return detached.to(torch.float32).numpy()


def from_tf(tf_dataset: Any) -> Source:
    """Build a `Source` from a ``tf.data.Dataset`` by materializing it to Arrow.

    Each element's tensors are converted to NumPy and concatenated column-wise;
    dict-structured elements keep their feature names as column names.
    """
    require("tensorflow", feature="TensorFlow interop", provides="tensorflow", extra="tensorflow")
    columns = _stack_tf_dataset(tf_dataset)
    return from_pydict(columns)


def from_spark(spark_df: Any) -> Source:
    """Build a `Source` from a Spark `DataFrame` via Arrow collection.

    Uses ``DataFrame.toArrow()`` (Spark 4+) when available, else the classic
    ``_collect_as_arrow``/``toPandas`` Arrow bridge. The collect is eager —
    Spark drives its own distributed read up to this boundary.
    """
    require("pyspark", feature="Spark interop", provides="PySpark", extra="spark")
    to_arrow = getattr(spark_df, "toArrow", None)
    if callable(to_arrow):
        return _source_from_table(to_arrow())
    return _source_from_table(pa.Table.from_pandas(spark_df.toPandas()))


def from_dask(ddf: Any) -> Source:
    """Build a streaming `Source` from a Dask `DataFrame`, one partition per batch.

    Returns an `IteratorSource` that computes one partition at a time (bounded
    memory), converting each pandas partition to an Arrow batch lazily.
    """
    require("dask", feature="Dask interop", provides="dask", extra="dask")
    schema = pa.Schema.from_pandas(ddf._meta)

    def _factory() -> Iterator[pa.RecordBatch]:
        for part in ddf.to_delayed():
            table = pa.Table.from_pandas(part.compute(), schema=schema)
            yield from table.to_batches()

    return IteratorSource(_factory, schema)


def from_ray_dataset(ray_dataset: Any) -> Source:
    """Build a streaming `Source` from a Ray Dataset, one Arrow block per batch.

    Ray is the migration on-ramp here: the dataset's Arrow blocks are iterated lazily
    into the engine (bounded memory), not collected to the driver. Ray stays a
    scheduling/transfer detail — bulk data does not round-trip the Ray object store.
    """
    require("ray", feature="Ray Dataset interop", provides="Ray", extra="ray")

    schema: pa.Schema | None = None
    for block in ray_dataset.iter_batches(batch_format="pyarrow"):
        schema = block.schema
        break
    if schema is None:  # empty dataset — fall back to Ray's reported schema
        schema = ray_dataset.schema().base_schema

    def _factory() -> Iterator[pa.RecordBatch]:
        for block in ray_dataset.iter_batches(batch_format="pyarrow"):
            yield from block.to_batches()

    return IteratorSource(_factory, schema)


# ---- helpers --------------------------------------------------------------
def numpy_to_column(ndarray: Any) -> pa.Array:
    """One Arrow column from a NumPy array whose leading axis is the row axis.

    The single place the rank rules live: 1-D is a scalar column, ``(n, dim)`` is a
    ``FixedSizeList`` (the embedding convention), and anything deeper is a
    fixed-shape-tensor column. :func:`from_numpy`, :func:`from_torch` and
    :func:`from_tf` all route through here so a per-row vector has the same type
    whichever door it came in by — and `api.session.frames` reaches it for the
    ``{name: array}`` spelling, so a ``(n, 384)`` embedding is one type through every door
    rather than a `FixedSizeList` through one and an error through another.

    A **masked** array keeps its mask. ``np.asarray`` drops the `np.ma` subclass, so a
    masked value used to arrive as whatever happened to sit under the mask — the fill,
    read as data, with no error and no warning. That is the one failure here that a
    correct-looking result hides, and it is why the mask is taken *before* any
    normalization: `np.ma` is how netCDF, h5py and ``np.genfromtxt`` spell a missing value.

    Args:
        ndarray: The array to convert; its first axis indexes rows.

    Returns:
        The Arrow column.

    Raises:
        PlanError: If the array's dtype or rank has no Arrow column form, naming the fix.
    """
    import numpy as np

    arr = ndarray if isinstance(ndarray, np.ndarray) else np.asarray(ndarray)
    mask = np.ma.getmaskarray(arr) if np.ma.isMaskedArray(arr) else None
    if mask is not None:
        arr = np.ma.getdata(arr)
        if not mask.any():
            mask = None
    if arr.dtype.names is not None:
        # A *nested* compound field — a C struct inside a record, which an h5py compound
        # dataset produces routinely. It is a struct column, not a table, because it sits
        # inside one: only the outermost compound dtype names the rows.
        fields = _field_columns(arr)
        return pa.StructArray.from_arrays(list(fields.values()), names=list(fields))
    _reject_untypable(arr, masked=mask is not None)
    if arr.ndim <= 1:
        return pa.array(arr, mask=mask)
    flat = np.ascontiguousarray(arr).reshape(-1)
    if arr.ndim == 2:
        child = pa.array(flat, mask=None if mask is None else np.ascontiguousarray(mask).ravel())
        return pa.FixedSizeListArray.from_arrays(child, arr.shape[1])
    from batcher.io.formats.ml.tensor import to_tensor_column

    return to_tensor_column(arr)


def _reject_untypable(arr: Any, *, masked: bool) -> None:
    """Refuse an array Arrow has no column type for, naming the conversion that fixes it.

    Every case here already failed; what it raised was a pyarrow message quoting an internal
    dtype *number* — ``Unsupported numpy type 15`` for a complex column, ``only handle
    1-dimensional arrays`` for a 0-d one. Neither names the column, the dtype, or anything to
    do about it, and both reach a user who wrote one ordinary line of NumPy.

    The three are declines rather than conversions on purpose. Arrow has no complex type, so
    any automatic split invents a column layout the caller did not ask for; a 0-d array has no
    row axis at all, so there is no row count to give it; and a rank >= 3 mask cannot ride
    along on a fixed-shape-tensor column, where filling it silently would be the very defect
    the mask handling above exists to prevent.
    """
    if arr.dtype.kind == "c":
        raise PlanError(
            f"a {arr.dtype} column has no Arrow type: Arrow does not represent complex "
            "numbers. Split it into two real columns first, e.g. "
            "`bt.from_numpy({'re': a.real, 'im': a.imag})`."
        )
    if arr.ndim == 0:
        raise PlanError(
            f"a 0-d array ({arr.dtype}) has no row axis, so it cannot become a column. "
            "Wrap it in one with `np.atleast_1d(a)` for a single-row column."
        )
    if masked and arr.ndim >= 3:
        raise PlanError(
            f"a masked {arr.ndim}-D array becomes a fixed-shape-tensor column, which has no "
            "place to record a per-element mask. Fill it first with `a.filled(0)`, or move "
            "the masked axis into its own column."
        )


def structured_to_columns(ndarray: Any) -> dict[str, pa.Array] | None:
    """A structured NumPy array as ``{field: column}``, or `None` when it is not one.

    A structured (record) array is NumPy's table: a 1-D array of a compound dtype, which is
    what ``np.genfromtxt``, ``np.rec.array``, an h5py compound dataset and most scientific
    binary readers produce. It reached Arrow as a single ``void`` column and failed with
    ``ArrowNotImplementedError: Unsupported numpy type 20`` — a message with neither the
    field names it holds nor the fact that they are what the caller wanted.

    One column per field is the reading every neighbor gives it (``pandas.DataFrame(rec)``,
    ``polars.from_numpy(rec)``), and each field goes back through `numpy_to_column`, so a
    sub-array field — ``dtype=[("emb", "f4", (384,))]`` — keeps the embedding convention
    instead of being flattened.

    Args:
        ndarray: The candidate array.

    Returns:
        The field columns in dtype order, or None when `ndarray` has no compound dtype.
    """
    import numpy as np

    if not isinstance(ndarray, np.ndarray) or ndarray.dtype.names is None:
        return None
    if ndarray.ndim != 1:
        raise PlanError(
            f"a structured array is a table, so its rows are its one axis; got "
            f"{ndarray.ndim}-D with fields {list(ndarray.dtype.names)}. Flatten it first "
            "with `a.reshape(-1)`."
        )
    return _field_columns(ndarray)


def _field_columns(arr: Any) -> dict[str, pa.Array]:
    """The columns of a structured array's fields, in dtype order.

    Split out so `numpy_to_column` can reach it for a *nested* compound field without going
    back through `structured_to_columns`, whose rank check is about the outermost array and
    whose `None` return would then be unreachable — a branch that exists only to be asserted
    away is worse than the one line it saves.
    """
    return {str(name): numpy_to_column(arr[name]) for name in arr.dtype.names}


def _column_from_rows(values: list[Any]) -> pa.Array:
    """One Arrow column from a list of per-row NumPy values, of any rank.

    Stacking straight into ``pa.array`` was the earlier spelling, and it raised
    ``ArrowInvalid: only handle 1-dimensional arrays`` for every feature that was not a
    scalar -- so ``from_tf`` accepted no vector feature at all, and ``from_torch``
    accepted one only when it arrived as a bare tensor rather than inside a dataset.
    """
    import numpy as np

    return numpy_to_column(np.stack(values))


def _stack_torch_dataset(dataset: Any, to_np: Any) -> dict[str, Any]:
    """Stack a map-style torch `Dataset` of tensor rows into Arrow columns."""
    rows = [dataset[i] for i in range(len(dataset))]
    if not rows:
        raise PlanError(
            "from_torch() needs a non-empty dataset: the column names and types are read "
            "off the first row, so an empty dataset carries no schema to build from."
        )
    return _stack_rows(rows, lambda v: _column_from_rows([to_np(r) for r in v]))


def _reject_batched_tf_dataset(tf_dataset: Any) -> None:
    """Refuse a batched ``tf.data.Dataset``, because one element must be one row.

    ``from_tf`` reads each element as a row, so a dataset that has been through
    ``.batch(n)`` would silently turn each *batch* into a single row holding a list --
    the values would all be present and every row count would be wrong. The batch axis
    is visible in ``element_spec`` as a leading ``None``, so it is caught here and the
    caller is pointed at ``.unbatch()`` rather than left with a reshaped result or, as
    before this check existed, a raw ``pyarrow.ArrowInvalid`` from deep inside the stack.
    """
    spec = getattr(tf_dataset, "element_spec", None)
    specs = (
        spec.values() if isinstance(spec, dict) else spec if isinstance(spec, tuple) else (spec,)
    )
    for one in specs:
        shape = getattr(one, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] is None:
            raise PlanError(
                "from_tf() reads one element as one row, so a batched tf.data.Dataset "
                "would make each batch a single row. Call .unbatch() on it first, or "
                "build the Dataset from the underlying arrays with bt.from_numpy()."
            )


def _stack_tf_dataset(tf_dataset: Any) -> dict[str, Any]:
    """Stack a ``tf.data.Dataset`` into Arrow columns via NumPy."""
    _reject_batched_tf_dataset(tf_dataset)
    rows = [_tf_element_to_np(el) for el in tf_dataset.as_numpy_iterator()]
    if not rows:
        raise PlanError(
            "from_tf() needs a non-empty dataset: the column names and types are read off "
            "the first element, so an empty dataset carries no schema to build from."
        )
    return _stack_rows(rows, _column_from_rows)


def _tf_element_to_np(element: Any) -> Any:
    """A ``tf.data`` element is already NumPy after ``as_numpy_iterator``."""
    return element


def _stack_rows(rows: list[Any], stack: Any) -> dict[str, Any]:
    """Stack a list of per-row elements (tensor / tuple / dict) into named columns."""
    first = rows[0]
    if isinstance(first, dict):
        return {k: stack([r[k] for r in rows]) for k in first}
    if isinstance(first, (tuple, list)):
        return {f"col_{i}": stack([r[i] for r in rows]) for i in range(len(first))}
    return {"data": stack(rows)}
