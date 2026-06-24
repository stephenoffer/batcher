"""The `bt.read` namespace — typed, per-format dataset readers.

`bt.read` is a single callable object: call it for path autodetection
(``bt.read("s3://b/*.parquet")``) or use a typed method per format
(``bt.read.parquet(path)``, ``bt.read.delta(uri, version=3)``,
``bt.read.kafka(topic="events")``). The methods are thin, typed wrappers over the
generic dispatch (`session.read`/`read_table`); format implementations live in
`io/formats/` and register into the `SOURCES` registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.api.session import read as _read
from batcher.api.session import read_table as _read_table

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["Reader", "read"]


def _decode(ds: Dataset, fn_name: str, **kwargs: Any) -> Dataset:
    """Append decoded media tensors to `ds` via the named `batcher.ml.decode` helper."""
    from batcher.ml import decode

    return getattr(decode, fn_name)(ds, **kwargs)


class Reader:
    """The `bt.read` namespace: callable for autodetect, typed methods per format.

    ``bt.read(path)`` infers the format from the URI scheme or file extension;
    ``bt.read.<format>(...)`` is the explicit, discoverable spelling. File/object
    formats take a path; catalog/SQL/NoSQL/streaming sources take their own
    connector arguments.
    """

    __slots__ = ()

    def __call__(self, path: str, *, format: str | None = None, **opts: Any) -> Dataset:
        """Read a file/object-store dataset, dispatching on `format` or the path.

        With no `format`, it is inferred from the URI scheme (``delta://``…) or the
        file extension. ``bt.read("s3://b/*.parquet")`` → Parquet;
        ``bt.read("data/", format="csv")``.
        """
        return _read(path, format=format, **opts)

    def table(self, format: str, *args: Any, **opts: Any) -> Dataset:
        """Read any registered non-file source by name (escape hatch).

        ``bt.read.table("delta", "s3://bucket/table", version=3)``. The typed
        methods below wrap this for the common backends.
        """
        return _read_table(format, *args, **opts)

    # --- File / object-store formats (path-addressed) ----------------------
    def parquet(self, path: str, **opts: Any) -> Dataset:
        """Read a Parquet file, directory, or glob (e.g. ``d/*.parquet``).

        Kyber pushes column projection and row-group predicates into the read, so a
        filtered/projected query touches only the needed columns and row groups.
        """
        return _read(path, format="parquet", **opts)

    def parquet_dataset(self, path: str, **opts: Any) -> Dataset:
        """Read a (Hive-)partitioned Parquet dataset directory.

        Partition columns are recovered from the directory layout, and projection plus
        predicate pushdown (including partition pruning) are applied per fragment.
        """
        return _read(path, format="parquet_dataset", **opts)

    def csv(self, path: str, **opts: Any) -> Dataset:
        """Read a CSV file, directory, or glob (e.g. ``d/*.csv``).

        The header row and column types are auto-inferred; column projection is pushed
        into the read and a single large file is split into newline-aligned byte ranges
        for parallel parsing.
        """
        return _read(path, format="csv", **opts)

    def json(self, path: str, **opts: Any) -> Dataset:
        """Read newline-delimited JSON: a file, directory, or glob.

        One JSON object per line; column types are inferred from the records.
        """
        return _read(path, format="json", **opts)

    def orc(self, path: str, **opts: Any) -> Dataset:
        """Read ORC file(s) — file, directory, or glob — with column projection pushed in."""
        return _read(path, format="orc", **opts)

    def arrow(self, path: str, **opts: Any) -> Dataset:
        """Read Arrow/Feather IPC file(s) — file, directory, or glob — zero-copy into the engine."""
        return _read(path, format="arrow", **opts)

    def avro(self, path: str, **opts: Any) -> Dataset:
        """Read Avro file(s): a file, directory, or glob.

        Needs the optional extra: ``pip install 'batcher-engine[avro]'``.
        """
        return _read(path, format="avro", **opts)

    def lance(self, path: str, **opts: Any) -> Dataset:
        """Read a Lance dataset (columnar ML format) by directory path.

        Needs the optional extra: ``pip install 'batcher-engine[lance]'``.
        """
        return _read(path, format="lance", **opts)

    def excel(self, path: str, **opts: Any) -> Dataset:
        """Read Excel workbook(s) — a file, directory, or glob — via python-calamine.

        Needs the optional extra: ``pip install 'batcher-engine[excel]'``.
        """
        return _read(path, format="excel", **opts)

    def xml(self, path: str, **opts: Any) -> Dataset:
        """Read XML file(s) — a file, directory, or glob — into columnar rows.

        Needs the optional extra: ``pip install 'batcher-engine[xml]'``.
        """
        return _read(path, format="xml", **opts)

    def logs(self, path: str, **opts: Any) -> Dataset:
        """Read line-delimited log file(s) as rows, one raw line per row by default.

        Pass ``pattern=`` to extract fields with a grok pattern instead.

        Args:
            path: A log file, directory, or glob.
            pattern: Optional grok pattern; named captures become columns.
        """
        return _read(path, format="logs", **opts)

    def text(self, path: str, **opts: Any) -> Dataset:
        """Read text file(s) as rows, one row per line by default.

        Args:
            path: A text file, directory, or glob.
            mode: ``"line"`` for one row per line, or ``"file"`` for whole-file rows.
        """
        return _read(path, format="text", **opts)

    def binary(self, path: str, **opts: Any) -> Dataset:
        """Read whole files as ``{uri, bytes, size, mime}`` rows.

        The entry point for custom/multimodal decoding of arbitrary file(s).
        """
        return _read(path, format="binary", **opts)

    def documents(self, path: str, **opts: Any) -> Dataset:
        """Read PDF document(s) — a file, directory, or glob — as extracted text rows.

        Needs the optional extra: ``pip install 'batcher-engine[pdf]'``.
        """
        return _read(path, format="documents", **opts)

    def numpy(self, path: str, **opts: Any) -> Dataset:
        """Read NumPy ``.npy``/``.npz`` file(s) — file, directory, or glob — as tensor rows."""
        return _read(path, format="numpy", **opts)

    def webdataset(self, path: str, **opts: Any) -> Dataset:
        """Read WebDataset ``.tar`` shard(s), grouping each sample's member files into one row."""
        return _read(path, format="webdataset", **opts)

    def hdf5(self, path: str, **opts: Any) -> Dataset:
        """Read HDF5 file(s) — a file, directory, or glob — with datasets as columns.

        Needs the optional extra: ``pip install 'batcher-engine[hdf5]'``.
        """
        return _read(path, format="hdf5", **opts)

    def zarr(self, path: str, **opts: Any) -> Dataset:
        """Read a Zarr store (chunked n-dimensional arrays) by path.

        Needs the optional extra: ``pip install 'batcher-engine[zarr]'``.
        """
        return _read(path, format="zarr", **opts)

    # --- Multimodal --------------------------------------------------------
    def images(
        self, path: str, *, decode: bool = False, size: tuple[int, int] | None = None, **opts: Any
    ) -> Dataset:
        """List image file(s) as ``{uri, bytes, size, mime}`` + header-metadata rows.

        ``decode=True`` (or passing ``size=``) appends an ``image`` (H, W, 3) uint8
        tensor column; decoding needs the optional extra:
        ``pip install 'batcher-engine[image]'``.

        Args:
            path: An image file, directory, or glob.
            decode: If true, append the decoded ``image`` tensor column.
            size: ``(height, width)`` to resize decoded images to; implies ``decode``.
        """
        ds = _read(path, format="images", **opts)
        return _decode(ds, "image_tensor_dataset", size=size) if (decode or size) else ds

    def audio(
        self, path: str, *, decode: bool = False, sample_rate: int | None = None, **opts: Any
    ) -> Dataset:
        """List audio file(s) + header-metadata rows.

        ``decode=True`` appends a ``waveform`` ``list<float32>`` column via soundfile,
        optionally resampled; decoding needs the optional extra:
        ``pip install 'batcher-engine[audio]'``.

        Args:
            path: An audio file, directory, or glob.
            decode: If true, append the decoded ``waveform`` column.
            sample_rate: Target sample rate in Hz to resample to when decoding.
        """
        ds = _read(path, format="audio", **opts)
        return _decode(ds, "audio_dataset", sample_rate=sample_rate) if decode else ds

    def video(
        self,
        path: str,
        *,
        decode: bool = False,
        size: tuple[int, int] | None = None,
        num_frames: int = 8,
        **opts: Any,
    ) -> Dataset:
        """List video file(s) + header-metadata rows.

        ``decode=True`` (or passing ``size=``) appends a ``frames`` (num_frames, H, W, 3)
        uint8 tensor column via PyAV; decoding needs the optional extra:
        ``pip install 'batcher-engine[video]'``.

        Args:
            path: A video file, directory, or glob.
            decode: If true, append the decoded ``frames`` tensor column.
            size: ``(height, width)`` to resize decoded frames to; implies ``decode``.
            num_frames: Number of frames to sample per video (default 8).
        """
        ds = _read(path, format="video", **opts)
        if not (decode or size):
            return ds
        return _decode(ds, "video_dataset", size=size, num_frames=num_frames)

    # --- Lakehouse ---------------------------------------------------------
    def delta(
        self,
        table_uri: str,
        *,
        version: int | None = None,
        timestamp: str | None = None,
        **opts: Any,
    ) -> Dataset:
        """Read a Delta Lake table by URI, defaulting to its latest version.

        Needs the optional extra: ``pip install 'batcher-engine[delta]'``.

        Args:
            table_uri: Path/URI of the Delta table root.
            version: Time-travel to this table version (exclusive with ``timestamp``).
            timestamp: Time-travel to the version current as of this ISO timestamp.
        """
        return _read_table("delta", table_uri, version=version, timestamp=timestamp, **opts)

    def iceberg(
        self,
        identifier: str,
        *,
        catalog: str | None = None,
        snapshot_id: int | None = None,
        **opts: Any,
    ) -> Dataset:
        """Read an Iceberg table by catalog identifier (e.g. ``"db.table"``).

        Needs the optional extra: ``pip install 'batcher-engine[iceberg]'``.

        Args:
            identifier: Table identifier within the catalog.
            catalog: Named catalog to resolve against (defaults to the configured one).
            snapshot_id: Time-travel to this Iceberg snapshot id.
        """
        return _read_table("iceberg", identifier, catalog=catalog, snapshot_id=snapshot_id, **opts)

    def hudi(self, table_uri: str, **opts: Any) -> Dataset:
        """Read an Apache Hudi table by URI (read-only, snapshot query).

        Needs the optional extra: ``pip install 'batcher-engine[hudi]'``.
        """
        return _read_table("hudi", table_uri, **opts)

    def delta_sharing(self, url: str, **opts: Any) -> Dataset:
        """Read a Delta Sharing table by ``<profile>#<share>.<schema>.<table>`` URL.

        Needs the optional extra: ``pip install 'batcher-engine[delta-sharing]'``.
        """
        return _read_table("delta_sharing", url, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read any ADBC/FlightSQL database in a single submission.

        Supply a SQL ``query`` positionally or ``table=`` to read a whole table; the
        connection is given via ``uri=`` / driver options.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.
        """
        return _read_table("adbc", query, **opts)

    def snowflake(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a Snowflake SQL query, fetching result chunks in parallel as Arrow.

        Connection credentials are passed as keyword options.
        """
        return _read_table("snowflake", query, **opts)

    def databricks(self, table: str, **opts: Any) -> Dataset:
        """Read a Databricks/Unity Catalog table by name.

        Uses credential vending to read the underlying Delta files directly.
        """
        return _read_table("databricks", table, **opts)

    def bigquery(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read BigQuery via the Storage Read API as parallel Arrow streams.

        Supply a SQL ``query`` positionally, or ``table=`` to read a whole table.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.
        """
        return _read_table("bigquery", query, **opts)

    def clickhouse(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a ClickHouse SQL query over the Arrow-native interface.

        Connection details are passed as keyword options.
        """
        return _read_table("clickhouse", query, **opts)

    # --- NoSQL -------------------------------------------------------------
    def mongo(self, **opts: Any) -> Dataset:
        """Read a MongoDB collection Arrow-natively via pymongoarrow.

        Pass connection, database, collection, and any query/projection as keyword options.
        """
        return _read_table("mongo", **opts)

    def cassandra(self, **opts: Any) -> Dataset:
        """Read a Cassandra/Scylla table, fanning out across token-range splits for parallelism.

        Pass connection, keyspace, and table as keyword options.
        """
        return _read_table("cassandra", **opts)

    def dynamodb(self, **opts: Any) -> Dataset:
        """Read a DynamoDB table using native parallel scan segments.

        Pass the table name and AWS connection options as keywords.
        """
        return _read_table("dynamodb", **opts)

    def elasticsearch(self, **opts: Any) -> Dataset:
        """Read an Elasticsearch index via ES|QL Arrow output (or a sliced scroll fallback).

        Pass the host, index, and query as keyword options.
        """
        return _read_table("elasticsearch", **opts)

    # --- Streaming ---------------------------------------------------------
    def kafka(self, **opts: Any) -> Dataset:
        """Read a Kafka topic as an unbounded streaming source.

        Pass ``topic=`` and broker/connection options as keywords; needs the optional
        extra: ``pip install 'batcher-engine[kafka]'``.
        """
        return _read_table("kafka", **opts)

    def kinesis(self, **opts: Any) -> Dataset:
        """Read an AWS Kinesis stream as an unbounded source.

        Pass the stream name and AWS options as keywords; needs the optional extra:
        ``pip install 'batcher-engine[kinesis]'``.
        """
        return _read_table("kinesis", **opts)

    def files_incremental(self, path: str, file_format: str, **opts: Any) -> Dataset:
        """Incrementally discover and read newly arrived files under `path`.

        A Databricks Auto Loader analog: tracks already-seen files across runs.

        Args:
            path: Directory or glob to watch for new files.
            file_format: Underlying format of those files (e.g. ``"parquet"``, ``"json"``).
        """
        return _read_table("files_incremental", path, file_format, **opts)

    def rate(self, rows_per_second: int = 1, **opts: Any) -> Dataset:
        """Generate ``(timestamp, value)`` rows at `rows_per_second` (Spark `rate`).

        A dev/benchmark source. Pass ``num_rows=`` to bound it (and ``pace=False`` to
        emit without the one-second cadence).
        """
        return _read_table("rate", rows_per_second, **opts)

    def socket(self, host: str = "localhost", port: int = 9999, **opts: Any) -> Dataset:
        """Read newline-delimited text from a TCP socket (Spark `socket`; dev only)."""
        return _read_table("socket", host, port, **opts)


read = Reader()
