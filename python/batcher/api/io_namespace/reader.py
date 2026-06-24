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
        r"""Read a file/object-store dataset, dispatching on `format` or the path.

        With no `format`, it is inferred from the URI scheme (``delta://``…) or the
        file extension. ``bt.read("s3://b/*.parquet")`` → Parquet;
        ``bt.read("data/", format="csv")``.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
                >>> _ = open(p, "w").write("a,b\n1,2\n")
                >>> bt.read(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format=format, **opts)

    def table(self, format: str, *args: Any, **opts: Any) -> Dataset:
        """Read any registered non-file source by name (escape hatch).

        ``bt.read.table("delta", "s3://bucket/table", version=3)``. The typed
        methods below wrap this for the common backends.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.table("delta", "s3://bucket/table", version=3)
        """
        return _read_table(format, *args, **opts)

    # --- File / object-store formats (path-addressed) ----------------------
    def parquet(self, path: str, **opts: Any) -> Dataset:
        """Read a Parquet file, directory, or glob (e.g. ``d/*.parquet``).

        Kyber pushes column projection and row-group predicates into the read, so a
        filtered/projected query touches only the needed columns and row groups.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> p = os.path.join(tempfile.mkdtemp(), "t.parquet")
                >>> pq.write_table(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.parquet(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="parquet", **opts)

    def parquet_dataset(self, path: str, **opts: Any) -> Dataset:
        """Read a (Hive-)partitioned Parquet dataset directory.

        Partition columns are recovered from the directory layout, and projection plus
        predicate pushdown (including partition pruning) are applied per fragment.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> root = os.path.join(tempfile.mkdtemp(), "pds")
                >>> pq.write_to_dataset(
                ...     pa.table({"x": [1, 2], "part": ["a", "b"]}),
                ...     root,
                ...     partition_cols=["part"],
                ... )
                >>> bt.read.parquet_dataset(root).select("x").sort("x").to_pydict()
                {'x': [1, 2]}
        """
        return _read(path, format="parquet_dataset", **opts)

    def csv(self, path: str, **opts: Any) -> Dataset:
        r"""Read a CSV file, directory, or glob (e.g. ``d/*.csv``).

        The header row and column types are auto-inferred; column projection is pushed
        into the read and a single large file is split into newline-aligned byte ranges
        for parallel parsing.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
                >>> _ = open(p, "w").write("a,b\n1,2\n")
                >>> bt.read.csv(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="csv", **opts)

    def json(self, path: str, **opts: Any) -> Dataset:
        r"""Read newline-delimited JSON: a file, directory, or glob.

        One JSON object per line; column types are inferred from the records.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.json")
                >>> _ = open(p, "w").write('{"a": 1, "b": 2}\n')
                >>> bt.read.json(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="json", **opts)

    def orc(self, path: str, **opts: Any) -> Dataset:
        """Read ORC file(s) — file, directory, or glob — with column projection pushed in.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.orc as orc
                >>> p = os.path.join(tempfile.mkdtemp(), "t.orc")
                >>> orc.write_table(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.orc(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="orc", **opts)

    def arrow(self, path: str, **opts: Any) -> Dataset:
        """Read Arrow/Feather IPC file(s) — file, directory, or glob — zero-copy into the engine.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> import pyarrow as pa, pyarrow.feather as fe
                >>> p = os.path.join(tempfile.mkdtemp(), "t.arrow")
                >>> fe.write_feather(pa.table({"a": [1], "b": [2]}), p)
                >>> bt.read.arrow(p).to_pydict()
                {'a': [1], 'b': [2]}
        """
        return _read(path, format="arrow", **opts)

    def avro(self, path: str, **opts: Any) -> Dataset:
        """Read Avro file(s): a file, directory, or glob.

        Needs the optional extra: ``pip install 'batcher-engine[avro]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.avro("data/events.avro")
        """
        return _read(path, format="avro", **opts)

    def lance(self, path: str, **opts: Any) -> Dataset:
        """Read a Lance dataset (columnar ML format) by directory path.

        Needs the optional extra: ``pip install 'batcher-engine[lance]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.lance("data/embeddings.lance")
        """
        return _read(path, format="lance", **opts)

    def excel(self, path: str, **opts: Any) -> Dataset:
        """Read Excel workbook(s) — a file, directory, or glob — via python-calamine.

        Needs the optional extra: ``pip install 'batcher-engine[excel]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.excel("report.xlsx")
        """
        return _read(path, format="excel", **opts)

    def xml(self, path: str, **opts: Any) -> Dataset:
        """Read XML file(s) — a file, directory, or glob — into columnar rows.

        Needs the optional extra: ``pip install 'batcher-engine[xml]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.xml("data/records.xml")
        """
        return _read(path, format="xml", **opts)

    def logs(self, path: str, **opts: Any) -> Dataset:
        """Read line-delimited log file(s) as rows, one raw line per row by default.

        Pass ``pattern=`` to extract fields with a grok pattern instead.

        Args:
            path: A log file, directory, or glob.
            pattern: Optional grok pattern; named captures become columns.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.logs(
                    "/var/log/app/*.log",
                    pattern="%{IP:client} %{WORD:method} %{URIPATHPARAM:path}",
                )
        """
        return _read(path, format="logs", **opts)

    def text(self, path: str, **opts: Any) -> Dataset:
        r"""Read text file(s) as rows, one row per line by default.

        Args:
            path: A text file, directory, or glob.
            mode: ``"line"`` for one row per line, or ``"file"`` for whole-file rows.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "t.txt")
                >>> _ = open(p, "w").write("hello\nworld\n")
                >>> bt.read.text(p).select("line_number", "text").to_pydict()
                {'line_number': [1, 2], 'text': ['hello', 'world']}
        """
        return _read(path, format="text", **opts)

    def binary(self, path: str, **opts: Any) -> Dataset:
        """Read whole files as ``{uri, bytes, size, mime}`` rows.

        The entry point for custom/multimodal decoding of arbitrary file(s).

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> p = os.path.join(tempfile.mkdtemp(), "b.bin")
                >>> _ = open(p, "wb").write(b"abc")
                >>> bt.read.binary(p).select("bytes", "size", "mime").to_pydict()
                {'bytes': [b'abc'], 'size': [3], 'mime': ['application/octet-stream']}
        """
        return _read(path, format="binary", **opts)

    def documents(self, path: str, **opts: Any) -> Dataset:
        """Read PDF document(s) — a file, directory, or glob — as extracted text rows.

        Needs the optional extra: ``pip install 'batcher-engine[pdf]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.documents("docs/*.pdf")
        """
        return _read(path, format="documents", **opts)

    def numpy(self, path: str, **opts: Any) -> Dataset:
        """Read NumPy ``.npy``/``.npz`` file(s) — file, directory, or glob — as tensor rows.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os, numpy as np
                >>> p = os.path.join(tempfile.mkdtemp(), "t.npy")
                >>> np.save(p, np.array([1, 2, 3]))
                >>> bt.read.numpy(p).to_pydict()
                {'data': [1, 2, 3]}
        """
        return _read(path, format="numpy", **opts)

    def webdataset(self, path: str, **opts: Any) -> Dataset:
        """Read WebDataset ``.tar`` shard(s), grouping each sample's member files into one row.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.webdataset("s3://bucket/shards/{000..099}.tar")
        """
        return _read(path, format="webdataset", **opts)

    def hdf5(self, path: str, **opts: Any) -> Dataset:
        """Read HDF5 file(s) — a file, directory, or glob — with datasets as columns.

        Needs the optional extra: ``pip install 'batcher-engine[hdf5]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.hdf5("data/measurements.h5")
        """
        return _read(path, format="hdf5", **opts)

    def zarr(self, path: str, **opts: Any) -> Dataset:
        """Read a Zarr store (chunked n-dimensional arrays) by path.

        Needs the optional extra: ``pip install 'batcher-engine[zarr]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.zarr("s3://bucket/array.zarr")
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

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.images("s3://bucket/images/*.jpg", decode=True, size=(224, 224))
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

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.audio("data/clips/*.wav", decode=True, sample_rate=16000)
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

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.video("s3://bucket/clips/*.mp4", decode=True, num_frames=16)
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
        stream: bool = False,
        starting_version: int = 0,
        **opts: Any,
    ) -> Dataset:
        """Read a Delta Lake table by URI, defaulting to its latest version.

        Needs the optional extra: ``pip install 'batcher-engine[delta]'``.

        Args:
            table_uri: Path/URI of the Delta table root.
            version: Time-travel to this table version (exclusive with ``timestamp``).
            timestamp: Time-travel to the version current as of this ISO timestamp.
            stream: Read the table as an unbounded stream of new commits (Spark
                ``readStream``) instead of a snapshot — see `delta_stream`. Requires
                ``delta.enableChangeDataFeed = true`` on the table.
            starting_version: When streaming, the first version to read from (default 0).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.delta("s3://bucket/delta/events", version=3)
        """
        if stream:
            return _read_table("delta_stream", table_uri, starting_version=starting_version, **opts)
        return _read_table("delta", table_uri, version=version, timestamp=timestamp, **opts)

    def read_change_feed(
        self, table_uri: str, *, starting_version: int = 0, **opts: Any
    ) -> Dataset:
        """Stream a Delta table's Change Data Feed (Databricks ``readChangeFeed``).

        Yields row-level changes — ``_change_type`` (insert/update/delete),
        ``_commit_version``, ``_commit_timestamp`` plus the data columns — for every
        commit after `starting_version`, as an unbounded source. Requires
        ``delta.enableChangeDataFeed = true`` on the table.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.read_change_feed("s3://bucket/delta/events", starting_version=10)
        """
        return _read_table(
            "delta_stream", table_uri, starting_version=starting_version, change_feed=True, **opts
        )

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

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.iceberg("db.events", catalog="prod")
        """
        return _read_table("iceberg", identifier, catalog=catalog, snapshot_id=snapshot_id, **opts)

    def hudi(self, table_uri: str, **opts: Any) -> Dataset:
        """Read an Apache Hudi table by URI (read-only, snapshot query).

        Needs the optional extra: ``pip install 'batcher-engine[hudi]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.hudi("s3://bucket/hudi/events")
        """
        return _read_table("hudi", table_uri, **opts)

    def delta_sharing(self, url: str, **opts: Any) -> Dataset:
        """Read a Delta Sharing table by ``<profile>#<share>.<schema>.<table>`` URL.

        Needs the optional extra: ``pip install 'batcher-engine[delta-sharing]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.delta_sharing("config.share#share.schema.table")
        """
        return _read_table("delta_sharing", url, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read any ADBC/FlightSQL database in a single submission.

        Supply a SQL ``query`` positionally or ``table=`` to read a whole table; the
        connection is given via ``uri=`` / driver options.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.sql(
                    "SELECT * FROM events WHERE country = 'US'",
                    uri="postgresql://localhost:5432/app",
                )
        """
        return _read_table("adbc", query, **opts)

    def snowflake(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a Snowflake SQL query, fetching result chunks in parallel as Arrow.

        Connection credentials are passed as keyword options.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.snowflake(
                    "SELECT * FROM sales.orders",
                    account="acme",
                    user="bob",
                    warehouse="wh",
                )
        """
        return _read_table("snowflake", query, **opts)

    def databricks(self, table: str, **opts: Any) -> Dataset:
        """Read a Databricks/Unity Catalog table by name.

        Uses credential vending to read the underlying Delta files directly.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.databricks("main.sales.orders")
        """
        return _read_table("databricks", table, **opts)

    def bigquery(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read BigQuery via the Storage Read API as parallel Arrow streams.

        Supply a SQL ``query`` positionally, or ``table=`` to read a whole table.

        Args:
            query: SQL text to execute, or ``None`` when reading via ``table=``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.bigquery("SELECT * FROM `project.dataset.events`")
        """
        return _read_table("bigquery", query, **opts)

    def clickhouse(self, query: str, **opts: Any) -> Dataset:
        """Read the result of a ClickHouse SQL query over the Arrow-native interface.

        Connection details are passed as keyword options.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.clickhouse("SELECT * FROM events", host="localhost")
        """
        return _read_table("clickhouse", query, **opts)

    # --- NoSQL -------------------------------------------------------------
    def mongo(self, **opts: Any) -> Dataset:
        """Read a MongoDB collection Arrow-natively via pymongoarrow.

        Pass connection, database, collection, and any query/projection as keyword options.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.mongo(
                    uri="mongodb://localhost:27017",
                    database="app",
                    collection="events",
                )
        """
        return _read_table("mongo", **opts)

    def cassandra(self, **opts: Any) -> Dataset:
        """Read a Cassandra/Scylla table, fanning out across token-range splits for parallelism.

        Pass connection, keyspace, and table as keyword options.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.cassandra(
                    contact_points=["127.0.0.1"],
                    keyspace="app",
                    table="events",
                )
        """
        return _read_table("cassandra", **opts)

    def dynamodb(self, **opts: Any) -> Dataset:
        """Read a DynamoDB table using native parallel scan segments.

        Pass the table name and AWS connection options as keywords.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.dynamodb(table="events", region="us-east-1")
        """
        return _read_table("dynamodb", **opts)

    def elasticsearch(self, **opts: Any) -> Dataset:
        """Read an Elasticsearch index via ES|QL Arrow output (or a sliced scroll fallback).

        Pass the host, index, and query as keyword options.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.elasticsearch(host="http://localhost:9200", index="events")
        """
        return _read_table("elasticsearch", **opts)

    # --- Streaming ---------------------------------------------------------
    def kafka(self, **opts: Any) -> Dataset:
        """Read a Kafka topic as an unbounded streaming source.

        Pass ``topic=`` and broker/connection options as keywords; needs the optional
        extra: ``pip install 'batcher-engine[kafka]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.kafka(topic="events", bootstrap_servers="localhost:9092")
        """
        return _read_table("kafka", **opts)

    def kinesis(self, **opts: Any) -> Dataset:
        """Read an AWS Kinesis stream as an unbounded source.

        Pass the stream name and AWS options as keywords; needs the optional extra:
        ``pip install 'batcher-engine[kinesis]'``.

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.kinesis(stream_name="events", region="us-east-1")
        """
        return _read_table("kinesis", **opts)

    def files_incremental(self, path: str, file_format: str, **opts: Any) -> Dataset:
        """Incrementally discover and read newly arrived files under `path`.

        A Databricks Auto Loader analog: tracks already-seen files across runs.

        Args:
            path: Directory or glob to watch for new files.
            file_format: Underlying format of those files (e.g. ``"parquet"``, ``"json"``).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.files_incremental("s3://bucket/incoming/", "parquet")
        """
        return _read_table("files_incremental", path, file_format, **opts)

    def rate(self, rows_per_second: int = 1, **opts: Any) -> Dataset:
        """Generate ``(timestamp, value)`` rows at `rows_per_second` (Spark `rate`).

        A dev/benchmark source. Pass ``num_rows=`` to bound it (and ``pace=False`` to
        emit without the one-second cadence).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.rate(rows_per_second=100, num_rows=1000, pace=False)
        """
        return _read_table("rate", rows_per_second, **opts)

    def socket(self, host: str = "localhost", port: int = 9999, **opts: Any) -> Dataset:
        """Read newline-delimited text from a TCP socket (Spark `socket`; dev only).

        Examples:
            .. code-block:: python

                import batcher as bt

                ds = bt.read.socket(host="localhost", port=9999)
        """
        return _read_table("socket", host, port, **opts)


read = Reader()
