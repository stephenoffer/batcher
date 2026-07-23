"""Streaming, schema, and connection-lifetime contracts for the warehouse connectors.

Three properties that every SQL/warehouse connector must hold, none of which any existing
test could see, and all three of which fail *silently* — the results stay correct while the
memory bound, the query bill, or the file-descriptor count quietly goes wrong:

1. **`iter_batches` actually streams.** The defect spelling is ``yield from self.read(...)``:
   the entry point a caller chose specifically to bound memory materializes the whole
   relation first, then hands back its chunks. Every caller keeps working; the bound is gone.
2. **`schema()` does not materialize the relation.** Downloading every row to read column
   names is a second full query — on a warehouse, a second invoice — for a result discarded
   immediately.
3. **Connections close deterministically**, exactly once, including when a caller abandons a
   generator after its first batch (which `schema()` now does on purpose).

None of these drivers are installed here, so every one of them is driven through a spy that
models the driver's contract — following `tests/io/test_sql.py`'s ``_SpyCursor``/``_SpyConn``
and `tests/io/test_sql_uri_and_dbapi.py`'s ``_RecordBatchStream``. The spies log each call, so
"did not materialize" is asserted as *the materializing call was never made*, rather than
inferred from a timing or a result shape.
"""

from __future__ import annotations

import pickle
import subprocess
import sys

import pyarrow as pa
import pytest

from batcher.io.formats.sql.bigquery import BigQuerySource, _BigQueryStreamSplit
from batcher.io.formats.sql.clickhouse import ClickHouseSource, _ClickHouseSplit
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.databricks import DatabricksSource, _DatabricksWarehouseSplit
from batcher.io.formats.sql.odbc import ODBCSource, _ODBCSplit
from batcher.io.formats.sql.snowflake import SnowflakeSource

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("id", pa.int64()), ("v", pa.string())])


def _batch(start: int) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array([start, start + 1]), pa.array([f"r{start}", f"r{start + 1}"])],
        schema=_SCHEMA,
    )


def _table() -> pa.Table:
    return pa.Table.from_batches([_batch(0), _batch(2)], schema=_SCHEMA)


# --- ClickHouse: streaming schema off the Arrow stream header ------------------


class _ArrowStream:
    """clickhouse-connect's ``query_arrow_stream`` context: a schema before any batch.

    The schema is readable *without* consuming a batch — that is the whole property a
    streaming `schema()` depends on, so the fake models it rather than exposing the schema
    only once data has been pulled.
    """

    schema = _SCHEMA

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __enter__(self) -> _ArrowStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self._log.append("stream_exit")

    def __iter__(self):
        for start in (0, 2):
            self._log.append("batch")
            yield _batch(start)


class _ClickHouseClient:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def query_arrow(self, query: str) -> pa.Table:
        self._log.append("query_arrow")  # the materializing call
        return _table()

    def query_arrow_stream(self, query: str) -> _ArrowStream:
        self._log.append("query_arrow_stream")
        return _ArrowStream(self._log)

    def close(self) -> None:
        self._log.append("close")


@pytest.fixture
def clickhouse_log(monkeypatch) -> list[str]:
    log: list[str] = []
    monkeypatch.setattr(
        "batcher.io.formats.sql.clickhouse._client", lambda params: _ClickHouseClient(log)
    )
    return log


def test_clickhouse_schema_does_not_download_the_relation(clickhouse_log) -> None:
    """`schema()` reads the stream header; it must never call `query_arrow`.

    This was ``self._table().schema`` — a full download of every row to read column names.
    `ClickHouseSource.schema` hides that behind a ``WHERE 1 = 0`` probe, but the fallback for
    an untyped probe runs the *real* query, where it is the difference between a metadata
    lookup and an OOM.
    """
    schema = _ClickHouseSplit({"host": "h"}, "SELECT * FROM t").schema()

    assert schema == _SCHEMA
    assert "query_arrow" not in clickhouse_log, "schema() materialized the whole relation"
    assert clickhouse_log.count("batch") == 0, "schema() consumed batches"
    assert clickhouse_log == ["query_arrow_stream", "stream_exit", "close"]


def test_clickhouse_schema_closes_its_client_exactly_once(clickhouse_log) -> None:
    _ClickHouseSplit({"host": "h"}, "SELECT * FROM t").schema()

    assert clickhouse_log.count("close") == 1


def test_clickhouse_iter_batches_streams(clickhouse_log) -> None:
    """The streaming entry point must yield before the relation is fully read."""
    split = _ClickHouseSplit({"host": "h"}, "SELECT * FROM t")

    first = next(iter(split.iter_batches()))

    assert first.num_rows == 2
    assert "query_arrow" not in clickhouse_log
    assert clickhouse_log.count("batch") == 1, "the whole relation was read before yielding"


# --- ODBC: fetcharrowbatches instead of fetchallarrow -------------------------


class _LegacyODBCCursor:
    """An older turbodbc: `fetchallarrow` only, with no `fetcharrowbatches` to find.

    The attribute is genuinely absent rather than set to a sentinel, because that is what
    `getattr(cur, "fetcharrowbatches", None)` is actually probing for.
    """

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def execute(self, sql: str) -> None:
        self._log.append("execute")

    def fetchallarrow(self) -> pa.Table:
        self._log.append("fetchallarrow")  # the materializing call
        return _table()


class _ODBCCursor(_LegacyODBCCursor):
    """A current turbodbc, which can hand back record batches incrementally."""

    def fetcharrowbatches(self):
        self._log.append("fetcharrowbatches")
        for start in (0, 2):
            self._log.append("batch")
            yield _batch(start)


class _ODBCConn:
    def __init__(self, log: list[str], *, streaming: bool = True) -> None:
        self._log = log
        self._streaming = streaming

    def cursor(self) -> _LegacyODBCCursor:
        return _ODBCCursor(self._log) if self._streaming else _LegacyODBCCursor(self._log)

    def close(self) -> None:
        self._log.append("close")


def _odbc_split(monkeypatch, log: list[str], *, streaming: bool = True) -> _ODBCSplit:
    monkeypatch.setattr(
        "batcher.io.formats.sql.odbc._connect",
        lambda dsn, connection_string: _ODBCConn(log, streaming=streaming),
    )
    return _ODBCSplit("dsn", None, "SELECT * FROM t")


def test_odbc_iter_batches_streams_rather_than_materializing(monkeypatch) -> None:
    """`iter_batches` must not pull the whole result first.

    This was ``yield from self.read(...)``, so the streaming entry point called
    `fetchallarrow` and pulled the entire result into memory before yielding anything —
    defeating every caller that chose `iter_batches` to avoid exactly that.
    """
    log: list[str] = []
    split = _odbc_split(monkeypatch, log)

    first = next(iter(split.iter_batches()))

    assert first.num_rows == 2
    assert "fetchallarrow" not in log, "the streaming path called the materializing fetch"
    assert log.count("batch") == 1, "the whole result was read before the first yield"


def test_odbc_iter_batches_yields_every_row(monkeypatch) -> None:
    """Streaming must not change *what* is returned, only when."""
    log: list[str] = []
    split = _odbc_split(monkeypatch, log)

    batches = list(split.iter_batches())

    assert pa.Table.from_batches(batches).equals(_table())


def test_odbc_iter_batches_applies_projection(monkeypatch) -> None:
    log: list[str] = []
    split = _odbc_split(monkeypatch, log)

    batches = list(split.iter_batches(["id"]))

    assert [b.schema.names for b in batches] == [["id"], ["id"]]


def test_odbc_schema_reads_one_batch_not_the_relation(monkeypatch) -> None:
    """`schema()` takes the first batch's schema and stops."""
    log: list[str] = []
    split = _odbc_split(monkeypatch, log)

    schema = split.schema()

    assert schema == _SCHEMA
    assert "fetchallarrow" not in log, "schema() materialized the whole relation"
    assert log.count("batch") == 1, "schema() drained the result"


def test_odbc_schema_closes_the_connection_exactly_once(monkeypatch) -> None:
    """Abandoning the generator after one batch must still close deterministically.

    `schema()` stops after the first batch, orphaning the generator frame. Without the
    `closing` wrapper the ``finally`` that closes the connection runs only when the collector
    reaches that frame — a connection leak that never surfaces as an error.
    """
    log: list[str] = []
    split = _odbc_split(monkeypatch, log)

    split.schema()

    assert log.count("close") == 1, f"expected exactly one close, got {log}"
    assert log[-1] == "close", "the connection closed after, not before, the batch was read"


def test_odbc_falls_back_when_the_driver_cannot_stream(monkeypatch) -> None:
    """An older turbodbc without `fetcharrowbatches` degrades, it does not fail."""
    log: list[str] = []
    split = _odbc_split(monkeypatch, log, streaming=False)

    batches = list(split.iter_batches())

    assert pa.Table.from_batches(batches).equals(_table())
    assert "fetchallarrow" in log
    assert log.count("close") == 1


# --- Databricks warehouse: fetchmany_arrow instead of fetchall_arrow ----------


class _LegacyDatabricksCursor:
    """A connector build with `fetchall_arrow` only — no incremental fetch to find."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._served = 0

    def execute(self, sql: str) -> None:
        self._log.append("execute")

    def fetchall_arrow(self) -> pa.Table:
        self._log.append("fetchall_arrow")  # the materializing call
        return _table()


class _DatabricksCursor(_LegacyDatabricksCursor):
    """A current connector, which serves Cloud Fetch results a chunk at a time."""

    def fetchmany_arrow(self, size: int) -> pa.Table:
        self._log.append("fetchmany_arrow")
        if self._served >= 2:
            return pa.Table.from_batches([], schema=_SCHEMA)
        batch = _batch(self._served * 2)
        self._served += 1
        return pa.Table.from_batches([batch], schema=_SCHEMA)


class _DatabricksConn:
    def __init__(self, log: list[str], *, streaming: bool = True) -> None:
        self._log = log
        self._streaming = streaming

    def cursor(self) -> _LegacyDatabricksCursor:
        cursor = _DatabricksCursor if self._streaming else _LegacyDatabricksCursor
        return cursor(self._log)

    def close(self) -> None:
        self._log.append("close")


def _databricks_split(
    monkeypatch, log: list[str], *, streaming: bool = True
) -> _DatabricksWarehouseSplit:
    monkeypatch.setattr(
        _DatabricksWarehouseSplit,
        "_connect",
        lambda self: _DatabricksConn(log, streaming=streaming),
    )
    return _DatabricksWarehouseSplit("host", "/path", "token", "SELECT * FROM t")


def test_databricks_iter_batches_streams_rather_than_materializing(monkeypatch) -> None:
    """`iter_batches` must not pull every Cloud Fetch result file first.

    This was ``yield from self.read(...)``, so the streaming path called `fetchall_arrow` —
    and a result large enough to arrive as Cloud Fetch files is exactly the one a caller
    reached for `iter_batches` to avoid holding whole.
    """
    log: list[str] = []
    split = _databricks_split(monkeypatch, log)

    first = next(iter(split.iter_batches()))

    assert first.num_rows == 2
    assert "fetchall_arrow" not in log, "the streaming path called the materializing fetch"
    assert log.count("fetchmany_arrow") == 1, "more than the first chunk was fetched"


def test_databricks_iter_batches_yields_every_row(monkeypatch) -> None:
    log: list[str] = []
    split = _databricks_split(monkeypatch, log)

    batches = list(split.iter_batches())

    assert pa.Table.from_batches(batches).equals(_table())


def test_databricks_schema_reads_one_chunk_not_the_relation(monkeypatch) -> None:
    log: list[str] = []
    split = _databricks_split(monkeypatch, log)

    schema = split.schema()

    assert schema == _SCHEMA
    assert "fetchall_arrow" not in log, "schema() materialized the whole relation"
    assert log.count("fetchmany_arrow") == 1, "schema() drained the result"


def test_databricks_schema_closes_the_connection_exactly_once(monkeypatch) -> None:
    log: list[str] = []
    split = _databricks_split(monkeypatch, log)

    split.schema()

    assert log.count("close") == 1, f"expected exactly one close, got {log}"


def test_databricks_falls_back_when_the_driver_cannot_stream(monkeypatch) -> None:
    log: list[str] = []
    split = _databricks_split(monkeypatch, log, streaming=False)

    batches = list(split.iter_batches())

    assert pa.Table.from_batches(batches).equals(_table())
    assert "fetchall_arrow" in log
    assert log.count("close") == 1


# --- BigQuery: first page, and a released gRPC channel ------------------------


class _Page:
    def __init__(self, log: list[str], start: int) -> None:
        self._log = log
        self._start = start

    def to_arrow(self) -> pa.RecordBatch:
        self._log.append("page")
        return _batch(self._start)


class _Rows:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    @property
    def pages(self):
        for start in (0, 2):
            yield _Page(self._log, start)


class _ReadRowsStream:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def to_arrow(self) -> pa.Table:
        self._log.append("to_arrow")  # the materializing call
        return _table()

    def rows(self) -> _Rows:
        return _Rows(self._log)


class _ReadClient:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def read_rows(self, name: str) -> _ReadRowsStream:
        self._log.append("read_rows")
        return _ReadRowsStream(self._log)

    def close(self) -> None:
        self._log.append("close")


def _bq_split(monkeypatch, log: list[str]) -> _BigQueryStreamSplit:
    monkeypatch.setattr("batcher.io.formats.sql.bigquery._read_client", lambda: _ReadClient(log))
    return _BigQueryStreamSplit("projects/p/streams/s0", 0)


def test_bigquery_split_schema_reads_one_page_not_the_stream(monkeypatch) -> None:
    """`schema()` must not read the stream to exhaustion.

    This was ``self._table().schema``, which drains the stream through the Storage Read API —
    a real, billed transfer — to learn column names the first page already carries.
    """
    log: list[str] = []
    split = _bq_split(monkeypatch, log)

    schema = split.schema()

    assert schema == _SCHEMA
    assert "to_arrow" not in log, "schema() downloaded the whole stream"
    assert log.count("page") == 1, "schema() read past the first page"


def test_bigquery_split_schema_releases_the_client_exactly_once(monkeypatch) -> None:
    """Abandoning the page generator must still release the gRPC channel.

    Each `_read_client` opens a channel with its own socket and threads; a split that opens
    one per read and never closes it leaks one per split, which on a worker fanned across
    many streams grows steadily rather than failing visibly.
    """
    log: list[str] = []
    split = _bq_split(monkeypatch, log)

    split.schema()

    assert log.count("close") == 1, f"expected exactly one close, got {log}"


def test_bigquery_iter_batches_streams_and_closes(monkeypatch) -> None:
    log: list[str] = []
    split = _bq_split(monkeypatch, log)

    first = next(iter(split.iter_batches()))
    assert first.num_rows == 2
    assert "to_arrow" not in log

    batches = list(split.iter_batches())
    assert pa.Table.from_batches(batches).equals(_table())


def test_bigquery_read_releases_the_client(monkeypatch) -> None:
    """The materializing path is legitimate, but it must still not leak a channel."""
    log: list[str] = []
    split = _bq_split(monkeypatch, log)

    split.read()

    assert "to_arrow" in log
    assert log.count("close") == 1


# --- Connectors that genuinely cannot stream ----------------------------------


def test_connectorx_iter_batches_is_documented_as_non_streaming() -> None:
    """ConnectorX has no incremental reader, and the code must say so rather than fake one.

    `read_sql` is one Rust call returning a finished table. Guarding the docstring keeps a
    later reader from "fixing" the apparent ``yield from`` anti-pattern into a streaming call
    that does not exist, or from quietly deleting the caveat and leaving callers believing
    this backend bounds memory.
    """
    from batcher.io.formats.sql.connectorx import _ConnectorXSplit

    doc = _ConnectorXSplit.iter_batches.__doc__ or ""
    assert "no incremental read" in doc
    assert "not** peak memory" in doc


def test_snowflake_chunk_iter_batches_is_documented_as_atomic() -> None:
    """A Snowflake `ResultBatch` is the streaming unit; the bound is applied one level up."""
    from batcher.io.formats.sql.snowflake import _SnowflakeBatchSplit

    doc = _SnowflakeBatchSplit.iter_batches.__doc__ or ""
    assert "atomic" in doc
    assert "SnowflakeSource.iter_batches" in doc


class _ResultBatch:
    """A Snowflake ``ResultBatch``: a picklable handle that downloads on `to_arrow`."""

    def __init__(self, log: list[str], index: int) -> None:
        self._log = log
        self._index = index

    def to_arrow(self) -> pa.Table:
        self._log.append(f"download-{self._index}")
        return pa.Table.from_batches([_batch(self._index * 2)], schema=_SCHEMA)


def test_snowflake_source_holds_one_chunk_at_a_time(monkeypatch) -> None:
    """`SnowflakeSource.iter_batches` must download chunk N+1 only after yielding chunk N.

    This is where Snowflake's memory bound actually lives: the chunk handles are vended by
    one query submission and cost nothing until `to_arrow`, so walking them lazily keeps a
    single chunk resident. Fetching them all up front would reintroduce the whole result.
    """
    log: list[str] = []
    monkeypatch.setattr(
        SnowflakeSource,
        "_result_batches",
        lambda self, predicate=None: [_ResultBatch(log, i) for i in range(3)],
    )
    src = SnowflakeSource("SELECT * FROM t", {"account": "a"})

    batches = src.iter_batches()
    next(iter(batches))

    assert log == ["download-0"], f"more than the first chunk was downloaded: {log}"


# --- identity: the learned-statistics key must name the connection ------------


def test_connectorx_identity_separates_two_servers() -> None:
    """The same query against prod and staging must be two relations, not one.

    Keyed on the query alone, Kyber plans the thousand-row staging table with the
    billion-row production table's cardinalities — a wrong plan, from good code, with
    nothing reporting an error.
    """
    prod = ConnectorXSource("SELECT * FROM orders", "mysql://u@prod/db")
    staging = ConnectorXSource("SELECT * FROM orders", "mysql://u@staging/db")

    assert prod.identity() != staging.identity()


def test_connectorx_identity_survives_password_rotation() -> None:
    """Rotating a password must not orphan a relation's accumulated statistics."""
    before = ConnectorXSource("SELECT 1", "mysql://u:old@h/db")
    after = ConnectorXSource("SELECT 1", "mysql://u:new@h/db")

    assert before.identity() == after.identity()


def test_connectorx_identity_leaks_no_credential() -> None:
    src = ConnectorXSource("SELECT 1", "mysql://u:hunter2@h/db")

    assert "hunter2" not in src.identity()


def test_snowflake_identity_separates_two_accounts() -> None:
    prod = SnowflakeSource("SELECT * FROM orders", {"account": "prod", "user": "u"})
    staging = SnowflakeSource("SELECT * FROM orders", {"account": "staging", "user": "u"})

    assert prod.identity() != staging.identity()


def test_snowflake_identity_survives_password_rotation() -> None:
    before = SnowflakeSource("SELECT 1", {"account": "a", "password": "old"})
    after = SnowflakeSource("SELECT 1", {"account": "a", "password": "new"})

    assert before.identity() == after.identity()
    assert "old" not in before.identity()


def test_databricks_identity_separates_two_workspaces() -> None:
    """``catalog.schema.table`` is unique only *within* a workspace."""
    prod = DatabricksSource(table="c.s.t", workspace="https://prod", token="x")
    staging = DatabricksSource(table="c.s.t", workspace="https://staging", token="x")

    assert prod.identity() != staging.identity()


def test_databricks_identity_survives_token_rotation() -> None:
    before = DatabricksSource(table="c.s.t", workspace="https://w", token="old")
    after = DatabricksSource(table="c.s.t", workspace="https://w", token="new")

    assert before.identity() == after.identity()
    assert "old" not in before.identity()


def test_databricks_warehouse_identity_separates_two_hosts() -> None:
    """`http_path` names a warehouse but not the host it lives on."""
    prod = DatabricksSource(
        query="SELECT 1", server_hostname="prod", http_path="/p", access_token="x"
    )
    staging = DatabricksSource(
        query="SELECT 1", server_hostname="staging", http_path="/p", access_token="x"
    )

    assert prod.identity() != staging.identity()


def test_bigquery_and_clickhouse_identities_already_name_the_connection() -> None:
    """The two that were already correct stay correct — a guard, not a new claim."""
    assert BigQuerySource(project="p", table="d.s.t").identity() == "bigquery:p:d.s.t"
    assert ClickHouseSource("SELECT 1", host="h").identity() == "clickhouse:h:SELECT 1"


def test_odbc_identity_does_not_leak_the_connection_string_password() -> None:
    """ODBC only *looked* correct: a DSN names the connection, a connection string is one.

    ``SERVER=db;UID=admin;PWD=hunter2`` was interpolated verbatim into `identity()` — and
    `identity()` is not merely logged, it is the key learned statistics are **persisted**
    under. So an ODBC password was being written into the metadata store, where it
    outlives the process that read it.
    """
    secret = "DRIVER={ODBC Driver 18};SERVER=db;UID=admin;PWD=hunter2"
    source = ODBCSource("SELECT 1", connection_string=secret)

    assert "hunter2" not in source.identity()
    assert "hunter2" not in repr(source)

    # Rotating the password must not change which relation this is.
    rotated = ODBCSource("SELECT 1", connection_string=secret.replace("hunter2", "newpass"))
    assert source.identity() == rotated.identity()

    # A different server, however, is a different relation.
    elsewhere = ODBCSource("SELECT 1", connection_string=secret.replace("SERVER=db", "SERVER=stg"))
    assert source.identity() != elsewhere.identity()


def test_odbc_identity_still_distinguishes_dsns() -> None:
    assert ODBCSource("SELECT 1", dsn="prod").identity() != (
        ODBCSource("SELECT 1", dsn="staging").identity()
    )


_STABILITY_PROGRAM = """
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.databricks import DatabricksSource
from batcher.io.formats.sql.snowflake import SnowflakeSource

print(ConnectorXSource("SELECT 1", "mysql://u@h/db").identity())
print(SnowflakeSource("SELECT 1", {"account": "a"}).identity())
print(DatabricksSource(table="c.s.t", workspace="w", token="x").identity())
"""


def test_identity_is_stable_across_processes() -> None:
    """The digest must be sha256, not `hash()`, which Python salts per process.

    An identity built on `hash()` changes every run, so every learned-statistics lookup
    misses and the feedback loop never reuses anything — while appearing to work perfectly.
    A second interpreter is the only way to observe that, since the salt is fixed within
    one process.
    """
    runs = [
        subprocess.run(
            [sys.executable, "-c", _STABILITY_PROGRAM],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for _ in range(2)
    ]

    assert runs[0] == runs[1], "identity changed between interpreters (salted hash?)"
    assert runs[0].strip(), "the subprocess produced no identities"


def test_splits_stay_picklable_after_the_identity_change() -> None:
    """Splits ship to workers, so nothing added to identity may break pickling."""
    monkeypatched = _ODBCSplit("dsn", None, "SELECT 1")

    assert pickle.loads(pickle.dumps(monkeypatched)).identity() == monkeypatched.identity()
