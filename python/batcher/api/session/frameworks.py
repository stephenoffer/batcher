"""Framework-interop constructors: a foreign object to a lazy `Dataset`.

One `from_<framework>` per ecosystem object a user is likely to be holding, plus
`from_any`, the type-dispatching entry point that migration code and `bt.sql`
bindings use so a caller never has to name the framework. Each adapter wraps the
`Source`-building function in `batcher.io.interop`, which stays `Dataset`-free so
no optional framework is pulled in at import time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.api.session._scan import _scan
from batcher.api.session.frames import (
    from_arrow,
    from_items,
    from_numpy,
    from_pydict,
    from_pylist,
)
from batcher.io import interop

__all__ = [
    "from_any",
    "from_dask",
    "from_duckdb",
    "from_huggingface",
    "from_pandas",
    "from_polars",
    "from_ray_dataset",
    "from_spark",
    "from_tf",
    "from_torch",
]


def from_pandas(df: Any) -> Dataset:
    """Create a `Dataset` from a pandas `DataFrame` via its Arrow bridge.

    Needs pandas (``pip install 'batcher-engine[pandas]'``); raises `BackendError`
    if it is absent. Goes through ``pyarrow.Table.from_pandas`` — no per-row Python.
    A pandas `Series` is accepted and becomes a one-column dataset.

    Examples:
        .. doctest::

            >>> import pandas as pd
            >>> import batcher as bt
            >>> bt.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]})).to_pydict()
            {'a': [1, 2], 'b': [3, 4]}

    Args:
        df: The pandas `DataFrame` (or `Series`) to ingest.

    Returns:
        A lazy `Dataset` over the frame.

    Raises:
        BackendError: If pandas is not installed.
    """
    if type(df).__name__ == "Series" and hasattr(df, "to_frame"):
        df = df.to_frame()
    return _scan(interop.from_pandas(df))


def from_polars(df: Any) -> Dataset:
    """Create a `Dataset` from a Polars `DataFrame` via its zero-copy Arrow export.

    Polars is Arrow-backed, so the buffers are referenced directly, not copied. A
    `LazyFrame` is collected first, and a `Series` becomes a one-column dataset.
    Needs polars (``pip install 'batcher-engine[polars]'``); raises `BackendError`
    if it is absent.

    Examples:
        .. doctest::

            >>> import polars as pl
            >>> import batcher as bt
            >>> bt.from_polars(pl.DataFrame({"a": [1, 2, 3]})).to_pydict()
            {'a': [1, 2, 3]}

    Args:
        df: The Polars `DataFrame`, `LazyFrame`, or `Series` to ingest.

    Returns:
        A lazy `Dataset` over the frame.

    Raises:
        BackendError: If polars is not installed.
    """
    if type(df).__name__ == "LazyFrame" and hasattr(df, "collect"):
        df = df.collect()
    if type(df).__name__ == "Series" and hasattr(df, "to_frame"):
        df = df.to_frame()
    return _scan(interop.from_polars(df))


def from_duckdb(source: Any, query: str | None = None) -> Dataset:
    """Create a `Dataset` from a DuckDB relation, connection, or in-process result.

    Pass a relation (``con.sql("SELECT ...")``) to hand its Arrow result straight
    over, or a connection plus `query` to run the statement first. The result is
    materialized through DuckDB's Arrow export, so nothing crosses row by row. To
    read a DuckDB *file* without an open connection use
    ``bt.read.sql(uri="duckdb:///path.db", query=...)``.

    Args:
        source: A DuckDB relation, or a DuckDB connection to run `query` on.
        query: The SQL to execute when `source` is a connection.

    Returns:
        A lazy `Dataset` over the query result.

    Raises:
        PlanError: If `source` is a connection and `query` is missing, or the object
            is neither a relation nor a connection.

    Examples:
        .. doctest::

            >>> import duckdb  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> rel = duckdb.sql("SELECT 1 AS x")  # doctest: +SKIP
            >>> bt.from_duckdb(rel).to_pydict()  # doctest: +SKIP
            {'x': [1]}
    """
    if query is not None:
        if not hasattr(source, "execute") and not hasattr(source, "sql"):
            raise PlanError(
                "from_duckdb(): a query needs a DuckDB connection as the first argument"
            )
        source = source.sql(query) if hasattr(source, "sql") else source.execute(query)
    for attr in ("arrow", "fetch_arrow_table", "to_arrow_table"):
        fetch = getattr(source, attr, None)
        if callable(fetch):
            return from_arrow(fetch())
    if hasattr(source, "__arrow_c_stream__"):
        return from_arrow(pa.table(source))
    raise PlanError(
        "from_duckdb() expects a DuckDB relation or result, got "
        f"{type(source).__name__}; pass a connection plus query= to run a statement"
    )


def from_huggingface(hf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a HuggingFace ``datasets.Dataset`` (Arrow-backed).

    HuggingFace datasets are Arrow tables underneath, so the table is taken
    directly. Needs ``datasets`` (``pip install 'batcher-engine[huggingface]'``);
    raises `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from datasets import load_dataset  # doctest: +SKIP
            >>> hf = load_dataset("imdb", split="train")  # doctest: +SKIP
            >>> ds = bt.from_huggingface(hf)  # doctest: +SKIP

    Args:
        hf_dataset: The HuggingFace ``datasets.Dataset`` to ingest.

    Returns:
        A lazy `Dataset` over the underlying Arrow table.

    Raises:
        BackendError: If ``datasets`` is not installed.
    """
    return _scan(interop.from_huggingface(hf_dataset))


def from_torch(dataset_or_tensors: Any) -> Dataset:
    """Create a `Dataset` from a PyTorch tensor, tuple of tensors, or `Dataset`.

    Tensors are moved to CPU and adapted via their NumPy buffers (one column per
    tensor); only bulk buffers cross into the engine, never per-row Python. Needs
    ``torch`` (``pip install 'batcher-engine[torch]'``); raises `BackendError` if
    it is absent.

    Examples:
        .. doctest::

            >>> import torch  # doctest: +SKIP
            >>> import batcher as bt
            >>> bt.from_torch(torch.tensor([1, 2, 3])).to_pydict()  # doctest: +SKIP
            {'data': [1, 2, 3]}

    Args:
        dataset_or_tensors: A PyTorch tensor, tuple of tensors, or ``Dataset``.

    Returns:
        A lazy `Dataset`, one column per tensor.

    Raises:
        BackendError: If ``torch`` is not installed.
    """
    return _scan(interop.from_torch(dataset_or_tensors))


def from_tf(tf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a ``tf.data.Dataset``, materializing it to Arrow.

    Each element's tensors are converted to NumPy and concatenated column-wise.
    Needs ``tensorflow`` (``pip install 'batcher-engine[tensorflow]'``); raises
    `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import tensorflow as tf  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> tf_ds = tf.data.Dataset.from_tensor_slices({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> ds = bt.from_tf(tf_ds)  # doctest: +SKIP

    Args:
        tf_dataset: The ``tf.data.Dataset`` to materialize.

    Returns:
        A lazy `Dataset` over the converted data.

    Raises:
        BackendError: If ``tensorflow`` is not installed.
    """
    return _scan(interop.from_tf(tf_dataset))


def from_spark(spark_df: Any) -> Dataset:
    """Create a `Dataset` from a Spark `DataFrame` by collecting it through Arrow.

    The Spark frame is collected to the driver via its Arrow bridge, so this
    materializes the data — for large frames write to a shared store and `read` it
    instead. Needs ``pyspark`` (``pip install 'batcher-engine[spark]'``); raises
    `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from pyspark.sql import SparkSession  # doctest: +SKIP
            >>> spark = SparkSession.builder.getOrCreate()  # doctest: +SKIP
            >>> sdf = spark.createDataFrame([(1,), (2,), (3,)], ["x"])  # doctest: +SKIP
            >>> ds = bt.from_spark(sdf)  # doctest: +SKIP

    Args:
        spark_df: The Spark `DataFrame` to collect through Arrow.

    Returns:
        A lazy `Dataset` over the collected data.

    Raises:
        BackendError: If ``pyspark`` is not installed.
    """
    return _scan(interop.from_spark(spark_df))


def from_dask(ddf: Any) -> Dataset:
    """Create a streaming `Dataset` from a Dask `DataFrame`, one partition per batch.

    Partitions stream lazily into the engine in bounded memory rather than being
    materialized at once. Needs ``dask`` (``pip install 'batcher-engine[dask]'``);
    raises `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import dask.dataframe as dd  # doctest: +SKIP
            >>> import pandas as pd  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> pdf = pd.DataFrame({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> ddf = dd.from_pandas(pdf, npartitions=2)  # doctest: +SKIP
            >>> ds = bt.from_dask(ddf)  # doctest: +SKIP

    Args:
        ddf: The Dask `DataFrame` to stream in, one partition per batch.

    Returns:
        A streaming lazy `Dataset` over the partitions.

    Raises:
        BackendError: If ``dask`` is not installed.
    """
    return _scan(interop.from_dask(ddf))


def from_ray_dataset(ray_dataset: Any) -> Dataset:
    """Create a streaming `Dataset` from a Ray Dataset (one Arrow block per batch).

    The migration on-ramp from Ray Data: blocks stream lazily into the engine in
    bounded memory. Requires `ray`.

    Examples:
        .. doctest::

            >>> import ray  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> rds = ray.data.range(100)  # doctest: +SKIP
            >>> ds = bt.from_ray_dataset(rds)  # doctest: +SKIP

    Args:
        ray_dataset: The Ray Dataset to stream in, one Arrow block per batch.

    Returns:
        A streaming lazy `Dataset` over the blocks.

    Raises:
        BackendError: If ``ray`` is not installed.
    """
    return _scan(interop.from_ray_dataset(ray_dataset))


# Foreign frame classes recognized by name, so dispatch never imports an optional
# framework just to test for it. Keyed by (module root, class name).
_BY_TYPE: dict[tuple[str, str], str] = {
    ("pandas", "DataFrame"): "from_pandas",
    ("pandas", "Series"): "from_pandas",
    ("polars", "DataFrame"): "from_polars",
    ("polars", "LazyFrame"): "from_polars",
    ("polars", "Series"): "from_polars",
    ("duckdb", "DuckDBPyRelation"): "from_duckdb",
    ("duckdb", "DuckDBPyConnection"): "from_duckdb",
    ("datasets", "Dataset"): "from_huggingface",
    ("torch", "Tensor"): "from_torch",
    ("tensorflow", "DatasetV2"): "from_tf",
    ("pyspark", "DataFrame"): "from_spark",
    ("dask", "DataFrame"): "from_dask",
    ("dask_expr", "DataFrame"): "from_dask",
    ("ray", "Dataset"): "from_ray_dataset",
    ("ray", "MaterializedDataset"): "from_ray_dataset",
}


def _dispatch_name(obj: Any) -> str | None:
    """The `from_*` function name for `obj`, matched on its type without importing it."""
    for cls in type(obj).__mro__:
        key = (cls.__module__.split(".", 1)[0], cls.__name__)
        if key in _BY_TYPE:
            return _BY_TYPE[key]
    return None


def from_any(data: Any) -> Dataset:
    """Create a `Dataset` from whatever object you are holding, dispatching on its type.

    The generic on-ramp, for scripts and migration code that should not have to name
    the framework: a `Dataset` passes through, a path string is `read`, an Arrow
    table/batch, dict, list of dicts, list of values, NumPy array, pandas or Polars
    frame, DuckDB relation, HuggingFace/Ray/Dask/Spark dataset, or anything exporting
    ``__arrow_c_stream__`` routes to the matching `from_*` constructor. Reach for the
    specific constructor when you know the type — the error messages are better.

    Args:
        data: The object to ingest.

    Returns:
        A lazy `Dataset` over `data`.

    Raises:
        PlanError: If no constructor matches the type of `data`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_any({"x": [1, 2]}).to_pydict()
            {'x': [1, 2]}
            >>> bt.from_any([{"x": 1}, {"x": 2}]).to_pydict()
            {'x': [1, 2]}
    """
    if isinstance(data, Dataset):
        return data
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        return from_arrow(data)
    if isinstance(data, str):
        from batcher.api.session.read import read

        return read(data)
    name = _dispatch_name(data)
    if name is not None:
        return globals()[name](data)
    if type(data).__module__.split(".", 1)[0] == "numpy":
        return from_numpy(data)
    if isinstance(data, Mapping):
        return from_pydict(data)
    if hasattr(data, "__arrow_c_stream__"):
        return from_arrow(data)
    if isinstance(data, Sequence):
        rows = list(data)
        if rows and all(isinstance(r, pa.RecordBatch) for r in rows):
            return from_arrow(rows)
        if rows and all(isinstance(r, Mapping) for r in rows):
            return from_pylist(rows)
        return from_items(rows)
    if isinstance(data, Iterable):
        from batcher.api.session.frames import from_iter

        return from_iter(data)
    raise PlanError(
        f"from_any(): no constructor for {type(data).__module__}.{type(data).__name__}. "
        "Convert it to an Arrow table, a dict of columns, or a list of row dicts first."
    )
