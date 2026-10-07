"""Warehouse write ergonomics, against fake client libraries injected into `sys.modules`.

Pins the request shapes each sink sends (BigQuery's Parquet load job with list inference,
Databricks' PUT / COPY INTO / REMOVE, Snowflake's ``write_pandas``), the job identity each
returns in `WrittenFile.job`, and the refusals made before anything is submitted.
"""

from __future__ import annotations

import io
import sys
import types
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.sql.snowflake import SnowflakeSink
from batcher.io.formats.sql.vendors import BigQuerySink, DatabricksSink
from batcher.io.formats.sql.vendors.bigquery_sink import check_bigquery_nesting
from batcher.io.manifest import WrittenFile
from batcher.io.sink import SINKS

pytestmark = pytest.mark.unit


# --- BigQuery ----------------------------------------------------------------------------


class _FakeBigQuery:
    """Records the client and load-job calls a `BigQuerySink` makes."""

    def __init__(self) -> None:
        self.loads: list[dict[str, Any]] = []
        self.module = types.ModuleType("google.cloud.bigquery")
        recorder = self

        class ParquetOptions:
            enable_list_inference = False

        class LoadJobConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class _Job:
            job_id = "job_123"
            location = "US"
            destination = "proj.ds.events"

            def __init__(self, rows: int) -> None:
                self.output_rows = rows

            def result(self) -> None:
                return None

        class Client:
            def __init__(self, project=None, location=None) -> None:
                self.project, self.location = project, location

            def load_table_from_file(self, fileobj, destination, job_config=None):
                table = pq.read_table(io.BytesIO(fileobj.read()))
                recorder.loads.append(
                    {
                        "client": self,
                        "destination": destination,
                        "config": job_config,
                        "table": table,
                    }
                )
                return _Job(table.num_rows)

        self.module.ParquetOptions = ParquetOptions
        self.module.LoadJobConfig = LoadJobConfig
        self.module.Client = Client
        self.module.SourceFormat = types.SimpleNamespace(PARQUET="PARQUET")
        self.module.WriteDisposition = types.SimpleNamespace(
            WRITE_APPEND="WRITE_APPEND", WRITE_TRUNCATE="WRITE_TRUNCATE"
        )


@pytest.fixture
def fake_bigquery(monkeypatch):
    fake = _FakeBigQuery()
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake.module)
    return fake


def test_bigquery_loads_parquet_with_list_inference_and_returns_the_job(fake_bigquery):
    table = pa.table({"id": [1, 2], "tags": [["a"], ["b", "c"]], "s": [{"x": 1}, {"x": 2}]})
    written = BigQuerySink(project="proj", location="EU").write(table, "proj.ds.events")
    (load,) = fake_bigquery.loads
    assert load["destination"] == "proj.ds.events"
    assert (load["client"].project, load["client"].location) == ("proj", "EU")
    config = load["config"]
    assert config.source_format == "PARQUET"
    assert config.write_disposition == "WRITE_APPEND"
    assert config.parquet_options.enable_list_inference is True
    # The nested shape travels intact: the list and the struct arrive as such.
    assert load["table"].column("tags").to_pylist() == [["a"], ["b", "c"]]
    assert pa.types.is_struct(load["table"].schema.field("s").type)
    assert written.rows == 2
    assert written.job == {
        "system": "bigquery",
        "job_id": "job_123",
        "location": "US",
        "destination": "proj.ds.events",
    }


def test_bigquery_overwrite_truncates_in_one_job(fake_bigquery):
    BigQuerySink(mode="overwrite").write(pa.table({"id": [1]}), "ds.t")
    assert fake_bigquery.loads[0]["config"].write_disposition == "WRITE_TRUNCATE"


def test_bigquery_refuses_a_distributed_overwrite(fake_bigquery):
    with pytest.raises(BackendError, match="truncate the table"):
        BigQuerySink(mode="overwrite").write_partitioned(
            pa.table({"id": [1]}), "ds.t", file_index=1
        )
    assert fake_bigquery.loads == []


def test_bigquery_refuses_an_unknown_mode():
    with pytest.raises(BackendError, match="upsert"):
        BigQuerySink(mode="upsert")


def test_bigquery_refuses_an_array_of_arrays_before_submitting(fake_bigquery):
    with pytest.raises(BackendError, match=r"column 'm' is an array of arrays"):
        BigQuerySink().write(pa.table({"m": [[[1, 2]], [[3]]]}), "ds.t")
    assert fake_bigquery.loads == []


def test_bigquery_refuses_a_null_array_element_naming_the_nested_path():
    table = pa.table({"s": [{"tags": ["a", None]}]})
    with pytest.raises(BackendError, match=r"column 's.tags' holds 1 NULL element"):
        check_bigquery_nesting(table)


def test_bigquery_accepts_a_null_array_itself():
    check_bigquery_nesting(pa.table({"tags": [["a"], None, []]}))


def test_ds_write_bigquery_reaches_the_sink(fake_bigquery):
    manifest = bt.from_pydict({"id": [1, 2, 3]}).write.bigquery("ds.t", project="proj")
    assert manifest.total_rows == 3
    assert manifest.files[0].job["job_id"] == "job_123"
    assert fake_bigquery.loads[0]["config"].write_disposition == "WRITE_APPEND"


def test_ds_write_bigquery_overwrite_is_passed_through(fake_bigquery):
    bt.from_pydict({"id": [1]}).write.bigquery("ds.t", mode="overwrite")
    assert fake_bigquery.loads[0]["config"].write_disposition == "WRITE_TRUNCATE"


# --- Databricks --------------------------------------------------------------------------


class _FakeDatabricks:
    """Records every connect and statement a `DatabricksSink` issues."""

    def __init__(self, fail_copy: bool = False) -> None:
        self.connects: list[dict[str, Any]] = []
        self.statements: list[str] = []
        self.staged: dict[str, bytes] = {}
        recorder = self

        class _Cursor:
            description = None
            query_id = None

            def execute(self, sql: str) -> None:
                recorder.statements.append(sql)
                self.query_id = f"q{len(recorder.statements)}"
                self.description = None
                if sql.startswith("PUT"):
                    local = sql.split("'")[1]
                    with open(local, "rb") as handle:
                        recorder.staged[sql.split("'")[3]] = handle.read()
                elif sql.startswith("COPY INTO"):
                    if fail_copy:
                        raise RuntimeError("schema mismatch")
                    self.description = [("num_affected_rows",), ("num_inserted_rows",)]

            def fetchall(self):
                return [(2, 2)]

        class _Conn:
            def cursor(self):
                return _Cursor()

            def close(self) -> None:
                pass

        def connect(**kwargs: Any) -> _Conn:
            recorder.connects.append(kwargs)
            return _Conn()

        self.module = types.ModuleType("databricks.sql")
        self.module.connect = connect


def _databricks_sink(**overrides: Any) -> DatabricksSink:
    kwargs = {
        "server_hostname": "adb.example",
        "http_path": "/sql/1.0/warehouses/w",
        "access_token": "tok",
        "volume_path": "/Volumes/main/staging/loads",
    }
    kwargs.update(overrides)
    return DatabricksSink(**kwargs)


def test_databricks_stages_copies_and_removes_with_a_fresh_name(monkeypatch):
    fake = _FakeDatabricks()
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    sink = _databricks_sink(catalog="main", db_schema="sales")
    table = pa.table({"id": [1, 2]})
    written = sink.write(table, "main.sales.order")
    put, copy, remove = fake.statements
    assert put.startswith("PUT '") and put.endswith("' OVERWRITE")
    remote = put.split("'")[3]
    assert remote.startswith("/Volumes/main/staging/loads/batcher-") and remote.endswith(".parquet")
    assert copy == f"COPY INTO `main`.`sales`.`order` FROM '{remote}' FILEFORMAT = PARQUET"
    assert remove == f"REMOVE '{remote}'"
    assert pq.read_table(io.BytesIO(fake.staged[remote])).equals(table)
    connect = fake.connects[0]
    assert connect["catalog"] == "main" and connect["schema"] == "sales"
    assert connect["staging_allowed_local_path"] in put
    assert written.rows == 2
    assert written.job["query_id"] == "q2"
    assert written.job["num_inserted_rows"] == 2
    # A second write must not reuse the name, or COPY INTO would skip it as already loaded.
    sink.write(table, "main.sales.order")
    assert fake.statements[3].split("'")[3] != remote


def test_databricks_failure_names_the_query_id_and_still_removes_the_file(monkeypatch):
    fake = _FakeDatabricks(fail_copy=True)
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    with pytest.raises(BackendError, match=r"query id q2.*schema mismatch"):
        _databricks_sink().write(pa.table({"id": [1]}), "t")
    assert fake.statements[-1].startswith("REMOVE '")


def test_databricks_refuses_overwrite_and_a_non_volume_path():
    with pytest.raises(BackendError, match="only 'append'"):
        _databricks_sink(mode="overwrite")
    with pytest.raises(BackendError, match="/Volumes/"):
        _databricks_sink(volume_path="dbfs:/tmp")


def test_databricks_token_is_not_in_the_repr():
    assert "tok" not in repr(_databricks_sink(access_token="tok-secret"))


class _RecordingSink:
    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = dict(kwargs)

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        return WrittenFile(path=path, rows=table.num_rows, bytes=0, job={"query_id": "q"})

    def commit(self, manifest: Any, path: str) -> None:
        return None


def test_ds_write_databricks_passes_session_options(monkeypatch):
    monkeypatch.setitem(SINKS._items, "databricks", _RecordingSink)
    manifest = bt.from_pydict({"id": [1]}).write.databricks(
        "t",
        volume_path="/Volumes/a/b/c",
        server_hostname="h",
        http_path="p",
        access_token="env:TOK",
        catalog="main",
        schema="sales",
    )
    kwargs = _RecordingSink.last_kwargs
    assert kwargs["mode"] == "append"
    assert kwargs["db_schema"] == "sales" and kwargs["catalog"] == "main"
    assert manifest.files[0].job == {"query_id": "q"}


# --- Snowflake ---------------------------------------------------------------------------


def _fake_snowflake(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    connector = types.ModuleType("snowflake.connector")

    class _Conn:
        def close(self) -> None:
            pass

    def connect(**kwargs: Any) -> _Conn:
        calls.append({"connect": kwargs})
        return _Conn()

    def write_pandas(conn, frame, table_name, auto_create_table, overwrite):
        calls.append({"table": table_name, "rows": len(frame), "overwrite": overwrite})
        return True, 1, len(frame), [("file0.parquet", "LOADED", len(frame), len(frame))]

    tools = types.ModuleType("snowflake.connector.pandas_tools")
    tools.write_pandas = write_pandas
    connector.connect = connect
    monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
    monkeypatch.setitem(sys.modules, "snowflake.connector", connector)
    monkeypatch.setitem(sys.modules, "snowflake.connector.pandas_tools", tools)
    return calls


def test_snowflake_write_returns_the_load_results(monkeypatch):
    calls = _fake_snowflake(monkeypatch)
    written = SnowflakeSink({"account": "a"}).write(pa.table({"id": [1, 2]}), "ORDERS")
    assert written.job == {
        "system": "snowflake",
        "chunks": 1,
        "rows_loaded": 2,
        "copy_into": [["file0.parquet", "LOADED", 2, 2]],
    }
    assert calls[1] == {"table": "ORDERS", "rows": 2, "overwrite": False}


def test_snowflake_browser_auth_is_refused_for_a_distributed_write(monkeypatch):
    _fake_snowflake(monkeypatch)
    sink = SnowflakeSink({"account": "a", "authenticator": "externalbrowser"}, mode="append")
    with pytest.raises(BackendError, match="no browser"):
        sink.write_partitioned(pa.table({"id": [1]}), "T", file_index=1)
    assert sink.write_partitioned(pa.table({"id": [1]}), "T", file_index=0)[0].rows == 1


def test_ds_write_snowflake_uses_the_declared_strategy(monkeypatch):
    calls = _fake_snowflake(monkeypatch)
    monkeypatch.setenv("SF_TOKEN", "t0ken")
    bt.from_pydict({"id": [1]}).write.snowflake(
        "ORDERS",
        mode="append",
        account="acme",
        user="etl",
        auth="oauth",
        token="env:SF_TOKEN",
        role="LOADER",
    )
    connect = calls[0]["connect"]
    assert connect["authenticator"] == "oauth"
    assert connect["role"] == "LOADER"
    assert connect["token"] == "t0ken"  # resolved only where the connection is opened
