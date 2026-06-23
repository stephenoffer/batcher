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
        """Read a Parquet file, directory, or glob (e.g. ``d/*.parquet``)."""
        return _read(path, format="parquet", **opts)

    def parquet_dataset(self, path: str, **opts: Any) -> Dataset:
        """Read a (Hive-)partitioned Parquet dataset directory."""
        return _read(path, format="parquet_dataset", **opts)

    def csv(self, path: str, **opts: Any) -> Dataset:
        """Read a CSV file, directory, or glob (e.g. ``d/*.csv``)."""
        return _read(path, format="csv", **opts)

    def json(self, path: str, **opts: Any) -> Dataset:
        """Read newline-delimited JSON: a file, directory, or glob."""
        return _read(path, format="json", **opts)

    def orc(self, path: str, **opts: Any) -> Dataset:
        """Read ORC file(s)."""
        return _read(path, format="orc", **opts)

    def arrow(self, path: str, **opts: Any) -> Dataset:
        """Read Arrow/Feather IPC file(s)."""
        return _read(path, format="arrow", **opts)

    def avro(self, path: str, **opts: Any) -> Dataset:
        """Read Avro file(s) (needs ``batcher-engine[avro]``)."""
        return _read(path, format="avro", **opts)

    def lance(self, path: str, **opts: Any) -> Dataset:
        """Read a Lance dataset (needs ``batcher-engine[lance]``)."""
        return _read(path, format="lance", **opts)

    def excel(self, path: str, **opts: Any) -> Dataset:
        """Read Excel workbook(s) (needs ``batcher-engine[excel]``)."""
        return _read(path, format="excel", **opts)

    def xml(self, path: str, **opts: Any) -> Dataset:
        """Read XML file(s) (needs ``batcher-engine[xml]``)."""
        return _read(path, format="xml", **opts)

    def logs(self, path: str, **opts: Any) -> Dataset:
        """Read line-delimited log file(s); pass ``pattern=`` for grok extraction."""
        return _read(path, format="logs", **opts)

    def text(self, path: str, **opts: Any) -> Dataset:
        """Read text file(s) as rows (``mode="line"`` or ``"file"``)."""
        return _read(path, format="text", **opts)

    def binary(self, path: str, **opts: Any) -> Dataset:
        """Read whole files as ``{uri, bytes, size, mime}`` rows."""
        return _read(path, format="binary", **opts)

    def documents(self, path: str, **opts: Any) -> Dataset:
        """Read PDF document(s) as text rows (needs ``batcher-engine[pdf]``)."""
        return _read(path, format="documents", **opts)

    def numpy(self, path: str, **opts: Any) -> Dataset:
        """Read NumPy ``.npy``/``.npz`` file(s)."""
        return _read(path, format="numpy", **opts)

    def webdataset(self, path: str, **opts: Any) -> Dataset:
        """Read WebDataset ``.tar`` shard(s)."""
        return _read(path, format="webdataset", **opts)

    def hdf5(self, path: str, **opts: Any) -> Dataset:
        """Read HDF5 file(s) (needs ``batcher-engine[hdf5]``)."""
        return _read(path, format="hdf5", **opts)

    def zarr(self, path: str, **opts: Any) -> Dataset:
        """Read a Zarr store (needs ``batcher-engine[zarr]``)."""
        return _read(path, format="zarr", **opts)

    # --- Multimodal --------------------------------------------------------
    def images(
        self, path: str, *, decode: bool = False, size: tuple[int, int] | None = None, **opts: Any
    ) -> Dataset:
        """List images (uri/bytes/size/mime + header meta); ``decode=True`` with
        ``size=(h, w)`` appends an ``image`` (H, W, 3) uint8 tensor (``batcher-engine[image]``)."""
        ds = _read(path, format="images", **opts)
        return _decode(ds, "image_tensor_dataset", size=size) if (decode or size) else ds

    def audio(
        self, path: str, *, decode: bool = False, sample_rate: int | None = None, **opts: Any
    ) -> Dataset:
        """List audio files + header meta; ``decode=True`` appends a ``waveform``
        ``list<float32>`` column via soundfile, optionally resampled (``batcher-engine[audio]``)."""
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
        """List video files + header meta; ``decode=True`` with ``size=(h, w)`` appends a
        ``frames`` (num_frames, H, W, 3) uint8 tensor via PyAV (``batcher-engine[video]``)."""
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
        """Read a Delta Lake table (time travel via ``version=``/``timestamp=``)."""
        return _read_table("delta", table_uri, version=version, timestamp=timestamp, **opts)

    def iceberg(
        self,
        identifier: str,
        *,
        catalog: str | None = None,
        snapshot_id: int | None = None,
        **opts: Any,
    ) -> Dataset:
        """Read an Iceberg table (``catalog=``, ``snapshot_id=``)."""
        return _read_table("iceberg", identifier, catalog=catalog, snapshot_id=snapshot_id, **opts)

    def hudi(self, table_uri: str, **opts: Any) -> Dataset:
        """Read an Apache Hudi table (read-only)."""
        return _read_table("hudi", table_uri, **opts)

    def delta_sharing(self, url: str, **opts: Any) -> Dataset:
        """Read a Delta Sharing table by profile URL."""
        return _read_table("delta_sharing", url, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read via ADBC/FlightSQL in a single submission (``query=`` or ``table=``)."""
        return _read_table("adbc", query, **opts)

    def snowflake(self, query: str, **opts: Any) -> Dataset:
        """Read a Snowflake query (parallel result-chunk fetch)."""
        return _read_table("snowflake", query, **opts)

    def databricks(self, table: str, **opts: Any) -> Dataset:
        """Read a Databricks/Unity Catalog table (credential vending → Delta)."""
        return _read_table("databricks", table, **opts)

    def bigquery(self, query: str | None = None, **opts: Any) -> Dataset:
        """Read BigQuery via the Storage Read API (parallel Arrow streams)."""
        return _read_table("bigquery", query, **opts)

    def clickhouse(self, query: str, **opts: Any) -> Dataset:
        """Read a ClickHouse query (Arrow-native)."""
        return _read_table("clickhouse", query, **opts)

    # --- NoSQL -------------------------------------------------------------
    def mongo(self, **opts: Any) -> Dataset:
        """Read a MongoDB collection (Arrow-native via pymongoarrow)."""
        return _read_table("mongo", **opts)

    def cassandra(self, **opts: Any) -> Dataset:
        """Read Cassandra/Scylla via token-range splits."""
        return _read_table("cassandra", **opts)

    def dynamodb(self, **opts: Any) -> Dataset:
        """Read DynamoDB via native parallel scan segments."""
        return _read_table("dynamodb", **opts)

    def elasticsearch(self, **opts: Any) -> Dataset:
        """Read Elasticsearch via ES|QL Arrow / sliced scroll."""
        return _read_table("elasticsearch", **opts)

    # --- Streaming ---------------------------------------------------------
    def kafka(self, **opts: Any) -> Dataset:
        """Read a Kafka topic as an unbounded streaming source."""
        return _read_table("kafka", **opts)

    def kinesis(self, **opts: Any) -> Dataset:
        """Read an AWS Kinesis stream as an unbounded source."""
        return _read_table("kinesis", **opts)

    def files_incremental(self, path: str, file_format: str, **opts: Any) -> Dataset:
        """Incrementally discover new files under `path` (Auto Loader analog)."""
        return _read_table("files_incremental", path, file_format, **opts)


read = Reader()
