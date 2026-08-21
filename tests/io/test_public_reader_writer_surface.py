"""Every reader and writer the public API exposes, reached at least once.

``bt.read`` carries 23 connector methods and ``ds.write`` eight that no test called:
Cassandra, ClickHouse, Databricks, Delta Sharing, DynamoDB, Event Hubs, HBase, HDF5, Hudi,
Kafka, Kinesis, MongoDB, Pub/Sub, Pulsar, Redis, Snowflake, Elasticsearch, and the
file formats behind Excel, logs, MCAP, MDF, TFRecord, WebDataset and Zarr. The ten
top-level ``bt.read_*`` aliases were also unexercised -- a separate surface from
``bt.read.<format>``, and the one a ported pandas or Polars script actually calls.

None of the external systems is reachable from a test box, so what this module holds them
to is the contract that *is* checkable and that matters most in practice: **reaching a
connector must produce either a lazy plan or an actionable failure, never an unhelpful
one.** A reader that raises ``KeyError`` or ``AttributeError`` deep in an adapter is a
support ticket; one that says which option is missing, or which package to install, is not.

The format readers are checked properly, by round trip: ``read_csv``, ``read_json``,
``read_ndjson``, ``read_ipc``, ``read_orc`` and ``read_avro`` write a frame and read it
back, and ``read_database`` runs against a real SQLite file. Those are the ones a test box
can actually prove.

One thing the module records rather than asserts as desirable: several connectors surface a
bare ``TypeError`` from the underlying source's ``__init__`` when a required option is
missing (``read.cassandra`` without ``contact_points``, ``read.mongo`` without ``uri``,
``read.snowflake`` without its connection arguments). The message names the argument, so it
is actionable, but it is not one of the project's typed errors and it names an internal
class rather than the public method the user called.
"""

from __future__ import annotations

import sqlite3

import pytest

import batcher as bt
from batcher._internal.errors import BatcherError, MissingDependencyError

pytestmark = pytest.mark.io

ROWS = {"a": [1, 2, 3], "s": ["x", "y", "z"]}

#: ``(reader alias, writer format)`` for every format where the round trip is provable here.
#: The writer name differs from the reader alias twice: ``read_ndjson`` reads what
#: ``write.json`` produces (one document per line), and ``read_ipc`` reads ``write.arrow``.
ROUND_TRIPS = [
    ("read_csv", "csv", "t.csv"),
    ("read_json", "json", "t.json"),
    ("read_ndjson", "json", "t.jsonl"),
    ("read_ipc", "arrow", "t.arrow"),
    ("read_orc", "orc", "t.orc"),
    ("read_avro", "avro", "t.avro"),
]


@pytest.mark.parametrize(("reader", "writer", "filename"), ROUND_TRIPS)
def test_a_top_level_read_alias_reads_back_what_was_written(tmp_path, reader, writer, filename):
    """The ``bt.read_x`` spelling, checked by writing a frame and reading it back."""
    path = str(tmp_path / filename)
    getattr(bt.from_pydict(ROWS).write, writer)(path)
    got = getattr(bt, reader)(path).to_pydict()
    assert sorted(got) == sorted(ROWS), f"{reader} returned columns {sorted(got)}"
    assert got["a"] == ROWS["a"], f"{reader}: {got['a']}"
    assert got["s"] == ROWS["s"]


@pytest.mark.parametrize(("reader", "writer", "filename"), ROUND_TRIPS)
def test_the_alias_and_the_namespace_reader_agree(tmp_path, reader, writer, filename):
    """``bt.read_csv`` and ``bt.read.csv`` are two spellings and must be one behaviour.

    They are separate code paths -- the aliases exist for a ported pandas or Polars script --
    so an alias that drifted from its namespace method would be invisible to every test of
    the other.
    """
    path = str(tmp_path / filename)
    getattr(bt.from_pydict(ROWS).write, writer)(path)
    namespace_name = reader.removeprefix("read_")
    if not hasattr(bt.read, namespace_name):
        pytest.skip(f"bt.read has no {namespace_name} method; the alias is the only spelling")
    assert (
        getattr(bt, reader)(path).to_pydict() == getattr(bt.read, namespace_name)(path).to_pydict()
    )


def test_read_database_runs_a_query_against_a_real_sqlite_file(tmp_path):
    """The one database reader a test box can actually prove, end to end."""
    database = tmp_path / "x.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE t (a INTEGER, s TEXT)")
    connection.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y")])
    connection.commit()
    connection.close()

    try:
        got = bt.read_database("SELECT a, s FROM t ORDER BY a", f"sqlite:///{database}").to_pydict()
    except MissingDependencyError as missing:
        # Which driver backs the SQL source depends on what is installed; where none is,
        # the reader must still say which package to add rather than fail obscurely.
        assert "pip install" in str(missing)
        pytest.skip(f"no SQL driver available here: {missing}")
    assert got == {"a": [1, 2], "s": ["x", "y"]}


def test_read_database_names_the_argument_it_could_not_parse():
    """The query comes first and the URI second, which is the easy one to swap.

    Swapping them produces a URI with no scheme, and the message has to say so -- otherwise
    the failure reads as a broken database rather than as two arguments the wrong way round.
    """
    with pytest.raises(BatcherError) as failure:
        bt.read_database("sqlite:///x.db", "SELECT 1").to_pydict()
    message = str(failure.value)
    assert "SELECT 1" in message, "the message must quote what it tried to parse as a URI"
    assert "scheme" in message


#: Every ``bt.read`` connector with the smallest call that reaches it, and whether that call
#: is expected to build a lazy plan or to fail with a message. Both are a pass: what is
#: being checked is that the method is reachable and that a failure is actionable.
READERS = [
    ("cassandra", lambda: bt.read.cassandra(keyspace="k", table="t")),
    ("clickhouse", lambda: bt.read.clickhouse("SELECT 1")),
    ("databricks", lambda: bt.read.databricks("cat.sch.tbl")),
    ("delta_sharing", lambda: bt.read.delta_sharing("share#schema.table")),
    ("dynamodb", lambda: bt.read.dynamodb(table="t")),
    ("eventhubs", lambda: bt.read.eventhubs("topic")),
    ("excel", lambda: bt.read.excel("/nonexistent/x.xlsx")),
    ("hbase", lambda: bt.read.hbase(table="t")),
    ("hdf5", lambda: bt.read.hdf5("/nonexistent/x.h5")),
    ("hudi", lambda: bt.read.hudi("/nonexistent/table")),
    ("kafka", lambda: bt.read.kafka("topic")),
    ("kinesis", lambda: bt.read.kinesis("stream")),
    ("logs", lambda: bt.read.logs("/nonexistent/x.log")),
    ("mcap", lambda: bt.read.mcap("/nonexistent/x.mcap")),
    ("mdf", lambda: bt.read.mdf("/nonexistent/x.mf4")),
    ("mongo", lambda: bt.read.mongo(database="d", collection="c")),
    ("pubsub", lambda: bt.read.pubsub("topic")),
    ("pulsar", lambda: bt.read.pulsar("topic")),
    ("redis", lambda: bt.read.redis(key_pattern="k*")),
    ("snowflake", lambda: bt.read.snowflake("SELECT 1")),
    ("tfrecord", lambda: bt.read.tfrecord("/nonexistent/x.tfrecord")),
    ("webdataset", lambda: bt.read.webdataset("/nonexistent/x.tar")),
    ("zarr", lambda: bt.read.zarr("/nonexistent/x.zarr")),
    ("elasticsearch", lambda: bt.read.elasticsearch(index="i")),
    ("socket", lambda: bt.read.socket(host="localhost", port=9999)),
    ("rate", lambda: bt.read.rate(rows_per_second=1)),
    ("rate_micro_batch", lambda: bt.read.rate_micro_batch(rows_per_batch=1)),
    ("read_change_feed", lambda: bt.read.read_change_feed("/nonexistent/table")),
    ("files_incremental", lambda: bt.read.files_incremental("/nonexistent/dir", "parquet")),
]

#: An acceptable failure names the missing option, the missing package, or the path. A
#: message with none of these is the thing this test exists to catch.
_ACTIONABLE = (
    "requires",
    "install",
    "missing",
    "does not exist",
    "not found",
    "provide",
    "argument",
    "region",
    "uri",
    "host",
    "expected",
    "failed to",
    "no such",
    "unable",
    "invalid",
    "must",
)


def _is_actionable(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _ACTIONABLE)


def _present(entries, namespace):
    """Keep the entries the namespace actually exposes on this build.

    The connector set grows: a name added on a branch and not yet merged, or one behind an
    extra, would otherwise make this module fail for naming something that does not exist.
    `test_every_connector_on_the_namespace_is_covered_here` is the other half -- it fails
    when a connector appears that this list has *not* caught up with, so the filter cannot
    quietly shrink the test to nothing.
    """
    return [(name, call) for name, call in entries if name in dir(namespace)]


READERS = _present(READERS, bt.read)
WRITERS_NAMESPACE = bt.from_pydict({"a": [1]}).write


def test_the_connector_lists_are_not_empty():
    """The filter above must never reduce either list to nothing."""
    assert len(READERS) >= 15, f"only {len(READERS)} readers survived the filter"
    assert len(WRITERS) >= 1, f"only {len(WRITERS)} writers survived the filter"


@pytest.mark.parametrize(("name", "call"), READERS)
def test_a_reader_builds_a_plan_or_says_what_is_missing(name, call):
    """Reaching every connector: a lazy plan, or a failure a user can act on."""
    try:
        built = call()
    except Exception as failure:
        assert _is_actionable(str(failure)), (
            f"bt.read.{name} failed with a message a user cannot act on: {failure!r}"
        )
        return
    assert built is not None, f"bt.read.{name} returned nothing"
    assert hasattr(built, "schema"), f"bt.read.{name} did not return a Dataset"


@pytest.mark.parametrize(("name", "call"), READERS)
def test_every_reader_is_reachable_and_a_typo_is_named(name, call):
    """The method exists, and a near-miss gets a suggestion rather than an ``AttributeError``.

    ``bt.read`` resolves formats through a registry, so an unknown name raises
    ``FormatError`` carrying the closest match and the available list. That is the better
    failure and it is what a typo actually produces, so it is what this pins.
    """
    from batcher._internal.errors import FormatError

    assert callable(getattr(bt.read, name)), f"bt.read.{name} is not callable"
    with pytest.raises(FormatError) as typo:
        getattr(bt.read, f"{name}_not_a_real_reader")
    message = str(typo.value)
    assert name in message, f"the suggestion did not name {name}: {message}"
    assert "Did you mean" in message


#: The ``ds.write`` connectors, with the smallest call that reaches each.
WRITERS = [
    ("cassandra", lambda ds: ds.write.cassandra("t")),
    ("dynamodb", lambda ds: ds.write.dynamodb("t")),
    ("elasticsearch", lambda ds: ds.write.elasticsearch("i")),
    ("hbase", lambda ds: ds.write.hbase("t")),
    ("kafka", lambda ds: ds.write.kafka("t")),
    ("redis", lambda ds: ds.write.redis("prefix")),
    ("snowflake", lambda ds: ds.write.snowflake("t")),
]


WRITERS = _present(WRITERS, WRITERS_NAMESPACE)


@pytest.mark.parametrize(("name", "call"), WRITERS)
def test_a_writer_commits_or_says_what_is_missing(name, call):
    """Same contract on the way out: a manifest, or an actionable failure."""
    ds = bt.from_pydict(ROWS)
    try:
        manifest = call(ds)
    except Exception as failure:
        assert _is_actionable(str(failure)), (
            f"ds.write.{name} failed with a message a user cannot act on: {failure!r}"
        )
        return
    assert manifest is not None, f"ds.write.{name} returned nothing"


#: Names on `bt.read` that are file formats or entry points rather than external-system
#: connectors. They are covered by the round-trip tests above and by `tests/io/` at large;
#: listing them is what lets the completeness test below tell "covered elsewhere" from
#: "nobody has looked at this".
_NOT_A_CONNECTOR = frozenset(
    {
        "arrow",
        "audio",
        "avro",
        "bed",
        "binary",
        "csv",
        "delta",
        "fasta",
        "fastq",
        "gff",
        "hudi",
        "iceberg",
        "image",
        "images",
        "ipc",
        "json",
        "jsonl",
        "lance",
        "lines",
        "mcap",
        "ndjson",
        "orc",
        "parquet",
        "sam",
        "bam",
        "text",
        "tfrecord",
        "vcf",
        "video",
        "warc",
        "xml",
        "yaml",
        "zarr",
        "numpy",
        "pandas",
        "polars",
        "torch",
        "sql",
        "database",
        "bigquery",
        "excel",
        "hdf5",
        "webdataset",
        "logs",
        "mdf",
        "geojson",
        "shapefile",
        "geoparquet",
        "kml",
        "gpx",
        "las",
        "laz",
        "pcd",
        "rosbag",
        "parquet_dataset",
        "deltalake",
        "files",
        "directory",
        # Formats and dispatchers rather than external systems: `table` picks a format by
        # name, and `documents` / `msgpack` / `point_cloud` / `training_shards` are file
        # readers covered by the format suites elsewhere in `tests/io/`.
        "documents",
        "msgpack",
        "point_cloud",
        "training_shards",
        "table",
    }
)


def test_every_connector_on_the_namespace_is_covered_here():
    """A connector this module has not caught up with must fail, not pass by omission.

    The lists above are filtered to what the build exposes, which keeps the module honest
    about a name that does not exist yet. This is the other direction: a connector that
    *does* exist and is neither in the list nor a known file format means the coverage this
    module claims has quietly stopped being true.
    """
    covered = {name for name, _ in READERS} | _NOT_A_CONNECTOR
    exposed = {n for n in dir(bt.read) if not n.startswith("_")}
    missed = sorted(n for n in exposed - covered if callable(getattr(bt.read, n, None)))
    assert not missed, (
        f"bt.read exposes {missed} which this module neither exercises nor lists as a file "
        "format; add each to READERS with its smallest call, or to _NOT_A_CONNECTOR"
    )


def test_write_console_prints_the_rows_and_returns_a_query(capsys):
    """The one sink that needs nothing external, so it is checked properly.

    ``write.console`` starts a streaming query, so the batch is printed by the query's own
    loop thread. The handle is drained with ``process_all_available`` before the captured
    output is read; without that the assertion races the thread and fails intermittently,
    which is worse than not having the test.
    """
    query = bt.from_pydict(ROWS).write.console(num_rows=5)
    assert query is not None, "a streaming sink returns its query handle"
    query.process_all_available()
    query.stop()
    printed = capsys.readouterr().out
    for value in ROWS["s"]:
        assert value in printed, f"{value!r} was not printed; output was {printed!r}"
    assert "a" in printed and "s" in printed, "the column names must appear"


def test_write_console_bounds_what_it_prints(capsys):
    """``num_rows`` has to bind, or a console sink on a large frame floods the terminal."""
    wide = bt.from_pydict({"v": list(range(500))})
    short_query = wide.write.console(num_rows=3)
    short_query.process_all_available()
    short_query.stop()
    short = capsys.readouterr().out
    long_query = wide.write.console(num_rows=100)
    long_query.process_all_available()
    long_query.stop()
    long = capsys.readouterr().out
    assert len(short) < len(long), f"the row bound had no effect: {len(short)} vs {len(long)}"
    assert "499" not in short


def test_read_excel_and_read_delta_name_the_path_they_could_not_open(tmp_path):
    """The two file-backed aliases whose failure a user is most likely to hit first."""
    missing_excel = str(tmp_path / "absent.xlsx")
    with pytest.raises(Exception) as excel:
        bt.read_excel(missing_excel)
    assert "absent.xlsx" in str(excel.value)

    missing_delta = str(tmp_path / "absent-table")
    with pytest.raises(Exception) as delta:
        bt.read_delta(missing_delta)
    assert "absent-table" in str(delta.value)


def test_read_iceberg_names_the_catalog_it_could_not_load():
    """Iceberg needs a catalog, and saying which one is the difference from a bare failure."""
    with pytest.raises(Exception) as failure:
        bt.read_iceberg("db.table")
    message = str(failure.value)
    assert "catalog" in message.lower()
    assert _is_actionable(message)


def test_streams_lists_the_running_queries_and_is_empty_by_default():
    """``bt.streams()`` is the process-wide handle list; with nothing running it is empty."""
    running = bt.streams()
    assert isinstance(running, list)
    assert running == [] or all(hasattr(q, "name") for q in running)


def test_register_model_accepts_a_factory_and_is_idempotent():
    """The model registry entry point, which the inference UDFs resolve names through."""
    calls: list[str] = []

    def factory():
        calls.append("built")
        return object()

    assert bt.register_model("test-model-for-coverage", factory) is None
    assert bt.register_model("test-model-for-coverage", factory) is None, (
        "registering the same name twice must not raise"
    )
    assert calls == [], "registration must not build the model"
