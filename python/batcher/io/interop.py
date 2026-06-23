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
Every other adapter does a deferred optional import and raises `BackendError`
with the right ``pip install 'batcher-engine[<extra>]'`` hint if the framework is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.source import InMemorySource, IteratorSource, Source

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "from_arrow",
    "from_dask",
    "from_huggingface",
    "from_numpy",
    "from_pandas",
    "from_polars",
    "from_pydict",
    "from_spark",
    "from_tf",
    "from_torch",
]


def _source_from_table(table: pa.Table) -> Source:
    """Wrap a (possibly empty) Arrow table as an `InMemorySource`.

    An empty table still has a schema, so it becomes a single empty batch — a
    source must always expose at least one batch to publish its schema.
    """
    batches = table.to_batches()
    if not batches:
        batches = [pa.RecordBatch.from_arrays([], schema=table.schema)]
    return InMemorySource(batches)


def _missing(framework: str, extra: str) -> BackendError:
    return BackendError(f"{framework} interop needs: pip install 'batcher-engine[{extra}]'")


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
        raise ValueError("from_arrow: empty batch list has no schema")
    return InMemorySource(batches)


def from_pydict(data: dict[str, Any]) -> Source:
    """Build a `Source` from a column-oriented ``{name: values}`` dict."""
    return _source_from_table(pa.table(data))


def from_numpy(ndarray: Any, *, column: str = "data") -> Source:
    """Build a single-column `Source` from a NumPy array under name `column`.

    The leading axis is the row axis. A 1-D array becomes a scalar column; an
    ``(n, dim)`` array becomes a ``FixedSizeList<…, dim>`` column (the embedding
    convention); an ``(n, *shape)`` array with ``shape`` of rank >= 2 becomes a
    fixed-shape-tensor column that preserves the full per-row shape.
    """
    import numpy as np

    arr = np.asarray(ndarray)
    if arr.ndim <= 1:
        col: pa.Array = pa.array(arr)
    elif arr.ndim == 2:
        dim = arr.shape[1]
        flat = pa.array(np.ascontiguousarray(arr).reshape(-1))
        col = pa.FixedSizeListArray.from_arrays(flat, dim)
    else:
        from batcher.io.formats.ml.tensor import to_tensor_column

        col = to_tensor_column(arr)
    return InMemorySource([pa.RecordBatch.from_arrays([col], names=[column])])


# ---- optional-framework adapters -----------------------------------------
def from_pandas(df: Any) -> Source:
    """Build a `Source` from a pandas `DataFrame` via ``pa.Table.from_pandas``."""
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise _missing("pandas", "pandas") from exc
    return _source_from_table(pa.Table.from_pandas(df))


def from_polars(df: Any) -> Source:
    """Build a `Source` from a Polars `DataFrame` via its zero-copy Arrow export."""
    try:
        import polars  # noqa: F401
    except ImportError as exc:
        raise _missing("polars", "polars") from exc
    return _source_from_table(df.to_arrow())


def from_huggingface(hf_dataset: Any) -> Source:
    """Build a `Source` from a HuggingFace `datasets.Dataset` (Arrow-backed).

    HuggingFace datasets are Arrow tables under the hood, so the underlying table
    is taken directly (zero-copy) — falling back to ``with_format('arrow')`` for
    dataset views that do not expose ``.data`` directly.
    """
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise _missing("huggingface", "huggingface") from exc
    data = getattr(hf_dataset, "data", None)
    table = getattr(data, "table", None)
    if isinstance(table, pa.Table):
        return _source_from_table(table)
    arrow_ds = hf_dataset.with_format("arrow")
    return _source_from_table(pa.Table.from_batches(list(arrow_ds.iter(batch_size=1024))))


def from_torch(dataset_or_tensors: Any) -> Source:
    """Build a `Source` from a PyTorch tensor, tuple of tensors, or `Dataset`.

    Tensors are moved to CPU and adapted via NumPy (one column per tensor); an
    iterable `Dataset` of tensor rows is stacked column-wise. No per-row Python
    crosses into the engine — only the bulk NumPy buffers do.
    """
    try:
        import torch
    except ImportError as exc:
        raise _missing("torch", "torch") from exc

    def _np(t: Any) -> Any:
        return t.detach().cpu().numpy()

    if isinstance(dataset_or_tensors, torch.Tensor):
        return from_numpy(_np(dataset_or_tensors))
    if isinstance(dataset_or_tensors, (tuple, list)) and all(
        isinstance(t, torch.Tensor) for t in dataset_or_tensors
    ):
        cols = {f"col_{i}": pa.array(_np(t)) for i, t in enumerate(dataset_or_tensors)}
        return from_pydict(cols)
    columns = _stack_torch_dataset(dataset_or_tensors, _np)
    return from_pydict(columns)


def from_tf(tf_dataset: Any) -> Source:
    """Build a `Source` from a ``tf.data.Dataset`` by materializing it to Arrow.

    Each element's tensors are converted to NumPy and concatenated column-wise;
    dict-structured elements keep their feature names as column names.
    """
    try:
        import tensorflow  # noqa: F401
    except ImportError as exc:
        raise _missing("tensorflow", "tensorflow") from exc
    columns = _stack_tf_dataset(tf_dataset)
    return from_pydict(columns)


def from_spark(spark_df: Any) -> Source:
    """Build a `Source` from a Spark `DataFrame` via Arrow collection.

    Uses ``DataFrame.toArrow()`` (Spark 4+) when available, else the classic
    ``_collect_as_arrow``/``toPandas`` Arrow bridge. The collect is eager —
    Spark drives its own distributed read up to this boundary.
    """
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:
        raise _missing("spark", "spark") from exc
    to_arrow = getattr(spark_df, "toArrow", None)
    if callable(to_arrow):
        return _source_from_table(to_arrow())
    return _source_from_table(pa.Table.from_pandas(spark_df.toPandas()))


def from_dask(ddf: Any) -> Source:
    """Build a streaming `Source` from a Dask `DataFrame`, one partition per batch.

    Returns an `IteratorSource` that computes one partition at a time (bounded
    memory), converting each pandas partition to an Arrow batch lazily.
    """
    try:
        import dask  # noqa: F401
    except ImportError as exc:
        raise _missing("dask", "dask") from exc
    schema = pa.Schema.from_pandas(ddf._meta)

    def _factory() -> Iterator[pa.RecordBatch]:
        for part in ddf.to_delayed():
            table = pa.Table.from_pandas(part.compute(), schema=schema)
            yield from table.to_batches()

    return IteratorSource(_factory, schema)


# ---- helpers --------------------------------------------------------------
def _stack_torch_dataset(dataset: Any, to_np: Any) -> dict[str, Any]:
    """Stack a map-style torch `Dataset` of tensor rows into Arrow columns."""
    import numpy as np

    rows = [dataset[i] for i in range(len(dataset))]
    if not rows:
        raise ValueError("from_torch: empty dataset has no schema")
    return _stack_rows(rows, lambda v: pa.array(np.stack([to_np(r) for r in v])))


def _stack_tf_dataset(tf_dataset: Any) -> dict[str, Any]:
    """Stack a ``tf.data.Dataset`` into Arrow columns via NumPy."""
    import numpy as np

    rows = [_tf_element_to_np(el) for el in tf_dataset.as_numpy_iterator()]
    if not rows:
        raise ValueError("from_tf: empty dataset has no schema")
    return _stack_rows(rows, lambda v: pa.array(np.stack(v)))


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
