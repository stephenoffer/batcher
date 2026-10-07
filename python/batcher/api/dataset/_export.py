"""Export helpers: `iter_batches` stream shaping and the hand-offs to other frames.

`shape_batches` turns `iter_batches`' consumer options (format, shuffle, ragged tail,
look-ahead) into one stream transform. The rest hand a result to NumPy/JAX arrays or to
another engine's frame: `to_ray_dataset`, `to_daft`, `to_spark`, `to_dask` and
`to_huggingface`. The framework-side conversion of the last two lives in `interop`; what
stays here is the part that knows about a `Dataset`: how its result is cut into partitions.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.plan.types import retained_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from batcher.api.dataset.frame import Dataset


def shape_batches(
    *,
    batch_size: int | None,
    batch_format: str,
    drop_last: bool,
    local_shuffle_buffer_size: int | None,
    local_shuffle_seed: int | None,
    prefetch_batches: int,
) -> Callable[[Iterator[pa.RecordBatch]], Iterator[Any]]:
    """Validate `iter_batches`' shaping options and return the stream transform they describe.

    Validation happens here, before the query runs, so a bad option names itself instead of
    surfacing from the first `next()` deep in a consumer. The transform then applies, in order:
    the local shuffle, the exact rebatch a shuffle or `drop_last` needs, the ragged tail, the
    format conversion, and the background look-ahead.

    Args:
        batch_size: The caller's rows per batch, or `None`.
        batch_format: The yielded type, one of `batcher.interop.formats.FORMATS`.
        drop_last: Drop a final batch shorter than `batch_size`.
        local_shuffle_buffer_size: Rows per shuffled block, or `None` for no shuffle.
        local_shuffle_seed: The shuffle seed; `None` means 0.
        prefetch_batches: Batches of background look-ahead; 0 disables it.

    Returns:
        A function from the engine's batch stream to the shaped stream.

    Raises:
        PlanError: If an option is invalid.
    """
    from batcher._internal.errors import PlanError
    from batcher.interop.formats import FORMATS

    if batch_format not in FORMATS:
        raise PlanError(
            f"iter_batches(batch_format=...) must be one of {sorted(FORMATS)}, got {batch_format!r}"
        )
    if not isinstance(drop_last, bool):
        raise PlanError(f"iter_batches(drop_last=...) must be a bool, got {drop_last!r}")
    if drop_last and batch_size is None:
        raise PlanError("iter_batches(drop_last=True) requires an explicit batch_size")
    for name, value, floor in (
        ("local_shuffle_buffer_size", local_shuffle_buffer_size, 1),
        ("local_shuffle_seed", local_shuffle_seed, 0),
        ("prefetch_batches", prefetch_batches, 0),
    ):
        if value is None and name != "prefetch_batches":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise PlanError(f"iter_batches({name}=...) must be an int >= {floor}, got {value!r}")

    def shape(batches: Iterator[pa.RecordBatch]) -> Iterator[Any]:
        from batcher._internal.prefetch import prefetch
        from batcher.api.terminal.stream.rebatch import _rebatch_exact

        stream: Iterator[Any] = batches
        if local_shuffle_buffer_size:
            stream = _shuffled_blocks(stream, local_shuffle_buffer_size, local_shuffle_seed or 0)
        if batch_size is not None and (local_shuffle_buffer_size or drop_last):
            stream = _rebatch_exact(stream, batch_size)
        if drop_last and batch_size is not None:
            stream = (b for b in stream if b.num_rows >= batch_size)
        if batch_format != "pyarrow":
            from batcher.interop.formats import to_format

            stream = (to_format(b, batch_format) for b in stream)
        return prefetch(stream, prefetch_batches)

    return shape


def _shuffled_blocks(
    batches: Iterator[pa.RecordBatch], buffer_rows: int, seed: int
) -> Iterator[pa.RecordBatch]:
    """Permute the rows within successive blocks of about `buffer_rows` rows.

    The blocks are the training loader's (`permuted_blocks`), including its byte ceiling, so
    a row count that is cheap over narrow rows cannot become an unbounded allocation over
    decoded images. Cutting a block early only narrows the shuffle window; no row is dropped
    or repeated.
    """
    from batcher.ml.loader.lazy import permuted_blocks

    for table, perm in permuted_blocks(batches, buffer_rows, seed):
        yield from table.take(pa.array(perm)).to_batches()


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


#: The `to_dask` materialization policies, in the order the docstring describes them.
DASK_POLICIES = ("arrow", "deferred", "parquet")

#: Target bytes per Dask partition under ``materialize="arrow"``. Dask's own guidance puts
#: a partition at around 100 MiB of pandas memory; Arrow is somewhat smaller than the pandas
#: frame it becomes, so this lands near it.
_DASK_PARTITION_BYTES = 128 * 1024 * 1024


def to_dask(
    ds: Dataset,
    *,
    materialize: str,
    npartitions: int | None,
    partition_bytes: int | None,
    staging_path: str | None,
) -> Any:
    """Hand the query's result to Dask as a lazy ``dask.dataframe.DataFrame``.

    See `Dataset.to_dask` for the three policies. None of them builds one pandas frame of
    the whole result: each partition becomes pandas inside its own Dask task.
    """
    from batcher._internal.errors import PlanError
    from batcher.interop.dask_frames import dask_from_arrow_partitions, import_dask_dataframe

    if materialize not in DASK_POLICIES:
        raise PlanError(
            f"to_dask(materialize={materialize!r}): expected one of {list(DASK_POLICIES)}"
        )
    if materialize == "parquet":
        dd = import_dask_dataframe()
        return dd.read_parquet(_stage_dataset(ds, staging_path, "to_dask"))
    if materialize == "deferred":
        n = _dask_partition_count(npartitions)
        return dask_from_arrow_partitions(range(n), _BucketLoader(ds, n), ds.schema)
    import_dask_dataframe()
    target = int(partition_bytes) if partition_bytes else _DASK_PARTITION_BYTES
    tables = list(_coalesced_tables(ds, None, target)) or [_empty_table(ds.schema)]
    return dask_from_arrow_partitions(tables, _identity, tables[0].schema)


def _identity(table: pa.Table) -> pa.Table:
    return table


def _dask_partition_count(npartitions: int | None) -> int:
    from batcher._internal.errors import PlanError

    n = npartitions if npartitions is not None else (os.cpu_count() or 1)
    if n < 1:
        raise PlanError(f"to_dask(npartitions={npartitions}): must be at least 1")
    return n


def _hashable_columns(schema: pa.Schema) -> list[str]:
    """The columns a content-hash bucket can be computed over: every non-nested one."""
    nested = (pa.types.is_nested, pa.types.is_null)
    return [f.name for f in schema if not any(check(f.type) for check in nested)]


class _BucketLoader:
    """One Dask partition of a deferred `to_dask`: the rows whose content hash is `part`.

    The bucket is a hash of the row's own values rather than a row number, because a
    parallel scan numbers rows differently on every run, and each partition is a separate
    run: a row-number split could hand a row to two partitions or to none. A content hash
    puts every row in exactly one bucket however the scan is scheduled.
    """

    def __init__(self, ds: Dataset, n: int) -> None:
        from batcher._internal.errors import PlanError

        columns = _hashable_columns(ds.schema)
        if not columns and n > 1:
            raise PlanError(
                "to_dask(materialize='deferred'): every column is nested, so there is no "
                "value to bucket rows by; use npartitions=1 or materialize='arrow'"
            )
        self._ds = ds
        self._n = n
        self._columns = columns

    def __call__(self, part: int) -> pa.Table:
        if self._n == 1:
            return self._ds.to_arrow()
        from batcher.plan.expr_ir import Col
        from batcher.plan.expr_ir.constructors import hash_rows

        digest = hash_rows(*(Col(c) for c in self._columns), seed=0)
        bucket = ((digest % self._n) + self._n) % self._n
        return self._ds.filter(bucket == part).to_arrow()


def _stage_dataset(ds: Dataset, staging_path: str | None, label: str) -> str:
    """Write the result as Parquet under a fresh directory and return that directory."""
    root = staging_path if staging_path is not None else tempfile.mkdtemp(prefix="batcher-")
    directory = f"{root.rstrip('/')}/{label}-{uuid.uuid4().hex}"
    ds.write.parquet(directory)
    return directory


#: `to_huggingface` modes.
HF_MODES = ("materialized", "iterable")


def to_huggingface(
    ds: Dataset,
    *,
    mode: str,
    class_labels: Any,
    images: Any,
) -> Any:
    """Hand the query's result to Hugging Face ``datasets``; see `Dataset.to_huggingface`."""
    from batcher._internal.errors import PlanError
    from batcher.interop.huggingface import HuggingFaceSpec, hf_dataset, hf_iterable_dataset

    if mode not in HF_MODES:
        raise PlanError(f"to_huggingface(mode={mode!r}): expected one of {list(HF_MODES)}")
    image_columns = [images] if isinstance(images, str) else list(images or ())
    spec = HuggingFaceSpec(
        ds.schema, class_labels=_label_names(ds, class_labels), images=image_columns
    )
    if mode == "iterable":
        return hf_iterable_dataset(ds.iter_batches, spec)
    return hf_dataset(ds.to_arrow(), spec)


def _label_names(ds: Dataset, class_labels: Any) -> dict[str, list[str]]:
    """``{column: names}`` from a mapping, or derived from each named column's values.

    A bare column name (or a list of them) means "the distinct non-null values, sorted",
    which costs one query per column; a mapping states the names and their code order.
    """
    if class_labels is None:
        return {}
    if isinstance(class_labels, str):
        class_labels = [class_labels]
    if isinstance(class_labels, Mapping):
        return {str(k): [str(v) for v in names] for k, names in class_labels.items()}
    from batcher._internal.errors import PlanError
    from batcher.plan.expr_ir import Col

    names: dict[str, list[str]] = {}
    for column in class_labels:
        index = ds.schema.get_field_index(column)
        if index < 0:
            continue  # HuggingFaceSpec names the missing column
        if pa.types.is_integer(ds.schema.field(index).type):
            raise PlanError(
                f"to_huggingface(): {column!r} holds integer codes, so its label names cannot "
                f"be derived from it; pass class_labels={{{column!r}: [name0, name1, ...]}}"
            )
        values = ds.select(column).filter(Col(column).is_not_null()).distinct().to_arrow()
        names[column] = sorted(str(v) for v in values.column(0).to_pylist())
    return names
