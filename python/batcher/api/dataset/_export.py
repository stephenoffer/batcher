"""Framework-export helpers behind `Dataset.to_torch` / `to_tf` and the engine hand-offs.

These bridge a `Dataset`'s output batches to PyTorch / TensorFlow training loops
via the `Dataset`-free converters in `batcher.ml.converters`. The batch source is
**re-iterable** — each pass re-runs the query — so a multi-epoch loader streams in
bounded memory rather than materializing the whole dataset. They also hand a result to
another engine's frame: `to_ray_dataset`, `to_daft` and `to_spark`.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import uuid
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.plan.types import retained_bytes

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from batcher.api.dataset.frame import Dataset


class _ReiterableBatches:
    """A re-iterable view over a dataset's output batches: each ``iter()`` re-runs
    the query, so a training framework can take multiple passes (epochs)."""

    __slots__ = ("_size", "_source")

    def __init__(self, source: Dataset, size: int | None) -> None:
        self._source = source
        self._size = size

    def __iter__(self) -> Any:
        return self._source.iter_batches(self._size)


def to_torch(ds: Dataset, columns: list[str] | None, batch_size: int | None) -> Any:
    """A re-iterable `torch.utils.data.IterableDataset` of per-batch tensor dicts."""
    from batcher.ml.converters import to_torch_iterable

    return to_torch_iterable(_ReiterableBatches(ds, batch_size), columns=columns)


def to_torch_dataloader(
    ds: Dataset, columns: list[str] | None, batch_size: int | None, **dl_kwargs: Any
) -> Any:
    """A `torch.utils.data.DataLoader` over the engine-batched tensor dicts.

    The engine already produces batches, so the loader uses ``batch_size=None``
    (one engine batch = one training batch); pass `batch_size` to size them.
    """
    from torch.utils.data import DataLoader

    return DataLoader(to_torch(ds, columns, batch_size), batch_size=None, **dl_kwargs)


def to_tf(ds: Dataset, columns: list[str] | None, batch_size: int | None) -> Any:
    """A re-iterable ``tf.data.Dataset`` of per-batch tensor dicts."""
    from batcher.ml.converters import to_tf_dataset

    return to_tf_dataset(_ReiterableBatches(ds, batch_size), columns=columns)


def to_jax(ds: Dataset, columns: list[str] | None) -> dict[str, Any]:
    """The full result as a ``{column: jax.Array}`` dict (needs ``jax``).

    The JAX counterpart of `to_numpy`: materializes each column as a NumPy array (tensor
    columns reshaped to ``(n, *shape)``) and wraps it with ``jax.numpy.asarray``. Raises
    `BackendError` if JAX is not installed.
    """
    jnp = require("jax.numpy", feature="Dataset.to_jax()", provides="JAX", extra="jax")
    return {name: jnp.asarray(arr) for name, arr in to_numpy(ds, columns).items()}


def to_numpy(ds: Dataset, columns: list[str] | None) -> dict[str, Any]:
    """The full result as a ``{column: numpy.ndarray}`` dict.

    Streams the output batches and concatenates each column, so a fixed-shape-tensor or
    fixed-size-list column (an image/embedding/feature-vector column) comes back as a real
    ``(n, *shape)`` array — not an opaque per-row object array — feeding NumPy / scikit-learn
    directly. Reuses `batcher.ml.to_numpy_batches` so the per-column conversion (tensor
    reshape, null→NaN, zero-copy where possible) matches the training-loader path exactly.
    """
    import numpy as np

    from batcher.ml.converters import to_numpy_batches

    names = list(ds.columns) if columns is None else list(columns)
    parts: dict[str, list[Any]] = {name: [] for name in names}
    for batch in to_numpy_batches(ds.iter_batches(), columns=names):
        for name in names:
            parts[name].append(batch[name])
    return {
        name: (np.concatenate(chunks) if chunks else np.array([])) for name, chunks in parts.items()
    }


#: Fallback target size for one Ray Data block, used when Ray's own
#: ``DataContext.target_max_block_size`` cannot be read. Matches Ray Data's documented
#: default so a Batcher-produced dataset blocks the same way a `read_parquet` one does.
_RAY_TARGET_BLOCK_BYTES = 128 * 1024 * 1024


def _ray_target_block_bytes() -> int:
    """Ray Data's configured target block size, or the documented default.

    Read from the live `DataContext` rather than hard-coded, so a cluster that tuned
    ``target_max_block_size`` gets blocks that match the rest of its pipeline. Ray moves
    this attribute between releases, so an unreadable context falls back rather than
    failing an export over a tuning knob.

    Anyscale's field guidance puts the usable band at **1 MiB to 128 MiB per block**, and
    Ray's own default sits at the top of it. That is the range a caller overriding
    `block_size_bytes` should stay inside: below it, per-block task overhead dominates and
    the dataset spends its time coalescing; above it, a single block stops fitting the
    object store's slack and spills.
    """
    try:
        from ray.data import DataContext

        size = getattr(DataContext.get_current(), "target_max_block_size", None)
        return int(size) if size else _RAY_TARGET_BLOCK_BYTES
    except Exception:  # pragma: no cover - Ray internals move between releases
        return _RAY_TARGET_BLOCK_BYTES


def _coalesced_tables(
    ds: Dataset, batch_size: int | None, block_bytes: int, distributed: bool | str = False
) -> Iterator[pa.Table]:
    """Yield the query's output as Arrow tables sized near `block_bytes`.

    Engine morsels (16,384 rows) are far smaller than a Ray Data block or a Daft partition,
    and a dataset of tens of thousands of tiny blocks schedules badly — per-block task
    overhead dominates and the receiving engine spends the job coalescing. Consecutive
    morsels are therefore accumulated until they reach the target, which is the same shape
    `read_parquet` produces. This generator holds at most one block at a time.
    """
    pending: list[pa.RecordBatch] = []
    pending_bytes = 0
    for batch in ds.iter_batches(batch_size, distributed=distributed):
        pending.append(batch)
        pending_bytes += retained_bytes(batch)
        if pending_bytes >= block_bytes:
            yield pa.Table.from_batches(pending)
            pending = []
            pending_bytes = 0
    if pending:
        yield pa.Table.from_batches(pending)


def to_ray_dataset(
    ds: Dataset,
    *,
    batch_size: int | None = None,
    block_size_bytes: int | None = None,
    distributed: bool | str = False,
) -> Any:
    """Hand the query's result to Ray Data as a `ray.data.Dataset`.

    The return leg of `bt.from_ray_dataset`: a Batcher result becomes the input of a Ray
    Train / Tune / Serve stage without a round trip through storage. Output batches are
    coalesced into Ray-sized Arrow blocks and put into the object store one block at a
    time, so the driver's footprint is one block rather than the whole result.
    """
    ray = require("ray", feature="Dataset.to_ray_dataset()", provides="Ray", extra="ray")
    require("ray.data", feature="Dataset.to_ray_dataset()", provides="Ray", extra="ray")

    target = int(block_size_bytes) if block_size_bytes else _ray_target_block_bytes()
    refs = [ray.put(block) for block in _coalesced_tables(ds, batch_size, target, distributed)]
    if not refs:
        # A zero-batch result still has a schema, and a Ray Dataset built from no blocks
        # has none — `.schema()` returns None and every downstream op fails on a column
        # the user can see in `ds.schema`. One empty block carries it across.
        refs = [ray.put(_empty_table(ds.schema))]
    return ray.data.from_arrow_refs(refs)


def _empty_table(schema: pa.Schema) -> pa.Table:
    """A zero-row table carrying `schema`, for handing an empty result to another engine."""
    return pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)


#: Target bytes per Daft partition built by `to_daft`. Daft sizes its own scan tasks between
#: ``scan_tasks_min_size_bytes`` (96 MiB) and ``scan_tasks_max_size_bytes`` (384 MiB) by
#: default, so a handed-over result lands inside the range a Daft read of it would produce.
_DAFT_PARTITION_BYTES = 128 * 1024 * 1024


def to_daft(ds: Dataset) -> Any:
    """Hand the query's result to Daft as a ``daft.DataFrame`` built from its Arrow batches.

    The output batches are coalesced into partition-sized tables and passed to
    ``daft.from_arrow``, which takes an iterable of tables. An empty result becomes one
    zero-row table, so the Daft frame still carries the schema.
    """
    daft = require("daft", feature="Dataset.to_daft()", provides="daft", extra="daft")
    tables = _coalesced_tables(ds, None, _DAFT_PARTITION_BYTES)
    first = next(tables, None)
    if first is None:
        return daft.from_arrow(_empty_table(ds.schema))
    return daft.from_arrow(itertools.chain([first], tables))


#: Results up to this many bytes reach Spark as one in-memory Arrow table; larger ones are
#: staged as Parquet. ``spark.createDataFrame`` serializes the whole table through the
#: driver to the JVM in one piece, so its cost grows with the result, while a Parquet read
#: is split and scanned by the executors.
_SPARK_ARROW_MAX_BYTES = 64 * 1024 * 1024


def to_spark(
    ds: Dataset, spark: Any, *, max_arrow_bytes: int | None, staging_path: str | None
) -> Any:
    """Hand the query's result to a Spark session as a ``pyspark.sql.DataFrame``.

    Batches are pulled until they pass `max_arrow_bytes`. A result that never does goes to
    ``spark.createDataFrame`` as one Arrow table. One that does is written, with the
    batches already pulled and the rest of the stream, to a fresh directory under
    `staging_path` as Parquet, and Spark reads that directory.
    """
    require("pyspark", feature="Dataset.to_spark()", provides="PySpark", extra="spark")
    limit = _SPARK_ARROW_MAX_BYTES if max_arrow_bytes is None else int(max_arrow_bytes)
    batches = ds.iter_batches()
    held: list[pa.RecordBatch] = []
    held_bytes = 0
    for batch in batches:
        held.append(batch)
        held_bytes += retained_bytes(batch)
        if held_bytes > limit:
            path = _stage_parquet(itertools.chain(held, batches), held[0].schema, staging_path)
            return spark.read.parquet(path)
    table = pa.Table.from_batches(held) if held else _empty_table(ds.schema)
    return spark.createDataFrame(_spark_arrow_input(table))


def _spark_arrow_input(table: pa.Table) -> Any:
    """`table` in the form this PySpark's ``createDataFrame`` accepts.

    PySpark 4 takes a `pyarrow.Table` directly and maps its types itself. PySpark 3 does
    not, so it gets the table through pandas, which is the Arrow path that release has.
    """
    import pyspark

    major = int(pyspark.__version__.split(".", 1)[0])
    return table if major >= 4 else table.to_pandas()


def _stage_parquet(
    batches: Iterable[pa.RecordBatch], schema: pa.Schema, staging_path: str | None
) -> str:
    """Write `batches` as one Parquet file in a new directory, returning the directory.

    The directory is new on every call, so Spark never reads a previous handoff's files
    alongside this one. Timestamps are written at microsecond precision, which is Spark's
    own, and a value that would lose precision raises rather than truncating silently.
    The staged files are not removed: Spark reads them lazily, for as long as the returned
    frame is used.
    """
    import pyarrow.parquet as pq
    from pyarrow import fs as pafs

    root = staging_path if staging_path is not None else tempfile.mkdtemp(prefix="batcher-")
    filesystem, base = pafs.FileSystem.from_uri(_as_uri(root))
    leaf = f"to_spark-{uuid.uuid4().hex}"
    directory = f"{base.rstrip('/')}/{leaf}"
    filesystem.create_dir(directory, recursive=True)
    with pq.ParquetWriter(
        f"{directory}/part-00000.parquet",
        schema,
        filesystem=filesystem,
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    ) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return f"{root.rstrip('/')}/{leaf}"


def _as_uri(path: str) -> str:
    """`path` as a URI pyarrow can resolve a filesystem from: local paths become absolute."""
    return path if "://" in path else os.path.abspath(path)
