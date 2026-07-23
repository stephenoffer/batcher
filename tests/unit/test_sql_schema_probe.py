"""`schema()` on a SQL source must not run the user's query.

Every relational connector answered `schema()` by executing the query in full and taking
`.schema` off the materialized Arrow table. Three things follow, none of them visible in a
test that reads a ten-row table:

* the **column names of a billion-row join cost the billion-row join**;
* the plan needs the schema *before* it executes, so an ordinary
  ``read(...).filter(...).collect()`` submits the whole query **twice** — the schema
  lookup, then the read;
* on a per-query or per-byte-billed warehouse (BigQuery, Databricks, Snowflake) the
  discarded first execution is a second real invoice.

Snowflake was worse still: it took `splits()[0]`, which *downloads a result chunk from
cloud storage*. BigQuery likewise opened and drained stream 0, when its
`create_read_session` response already carries the Arrow schema for free. Both also
indexed `[0]` unguarded, so an empty relation — which still has columns — raised
`IndexError`.

These tests use stub drivers, because the point is exactly *what reaches the backend*, and
that is unobservable through a real connection. Each stub records the SQL it is handed and
returns a row count that depends on it, so "did the schema lookup scan the table" becomes
an assertion rather than an inference.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit

_BIG = 5_000_000
_SCHEMA = pa.schema([("k", pa.int64()), ("v", pa.string())])


def _result(sql: str) -> pa.Table:
    """A typed result whose size depends on whether the probe's ``WHERE 1 = 0`` is present.

    Typed even when empty — that is what every real backend does, since result-set
    metadata comes from the query, not from the rows.
    """
    rows = 0 if "1 = 0" in sql else _BIG
    return pa.table(
        {"k": pa.array(range(rows), pa.int64()), "v": pa.array(["x"] * rows, pa.string())}
    )


class _ArrowStream:
    """A stub for ClickHouse's Arrow stream: a schema up front, batches only on demand.

    Modeling the schema as available *without* iterating is the whole point — a real
    Arrow IPC stream carries it in the header, which is why reading a schema off the
    stream costs no rows. A fake that only exposed the schema after materializing would
    make the streaming path untestable and quietly bless the behavior it replaced.
    """

    def __init__(self, table: pa.Table, on_batch) -> None:
        self._table = table
        self._on_batch = on_batch

    schema = property(lambda self: self._table.schema)

    def __enter__(self) -> _ArrowStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):
        for batch in self._table.to_batches():
            self._on_batch(batch.num_rows)
            yield batch


class _Recorder:
    """A stub ClickHouse client that records every statement it is asked to run."""

    def __init__(self) -> None:
        self.sql: list[str] = []
        #: Rows actually pulled off a stream, as opposed to rows the statement *could*
        #: have returned. A streamed `schema()` reads zero.
        self.streamed_rows = 0

    def query_arrow(self, sql: str) -> pa.Table:
        self.sql.append(sql)
        return _result(sql)

    def query_arrow_stream(self, sql: str) -> _ArrowStream:
        self.sql.append(sql)

        def _count(rows: int) -> None:
            self.streamed_rows += rows

        return _ArrowStream(_result(sql), _count)

    def close(self) -> None:
        pass

    @property
    def rows_scanned(self) -> int:
        return sum(_result(s).num_rows for s in self.sql)


@pytest.fixture
def clickhouse(monkeypatch):
    """A ClickHouse source wired to a recording stub.

    ClickHouse stands in for the family: `_common.schema_probe` is shared, and the five
    query-based connectors (ADBC, ODBC, ClickHouse, ConnectorX, Databricks) use it
    identically. Testing the shared helper plus one wiring is the honest coverage —
    asserting the other four through their own stubs would test the stubs.
    """
    from batcher.io.formats.sql import clickhouse as mod

    recorder = _Recorder()
    monkeypatch.setattr(mod, "_client", lambda params: recorder)
    return mod.ClickHouseSource(query="SELECT * FROM big", host="h"), recorder


def test_schema_does_not_scan_the_table(clickhouse) -> None:
    """The headline: this used to materialize five million rows to read two names."""
    source, recorder = clickhouse

    source.schema()

    assert recorder.rows_scanned == 0, f"schema() scanned {recorder.rows_scanned} rows"
    # Two independent guarantees, and the second is the newer one: the *statement* asks
    # for no rows (`WHERE 1 = 0`), and the schema is read off the stream header so no row
    # is pulled even when the statement would have returned some. Either alone leaves the
    # fallback path — which runs the real query — free to download a whole table.
    assert recorder.streamed_rows == 0, f"schema() streamed {recorder.streamed_rows} rows"


def test_schema_is_still_correct(clickhouse) -> None:
    """Cheap and wrong would be worse than expensive and right."""
    source, _ = clickhouse

    assert source.schema() == _SCHEMA


def test_the_probe_is_the_zero_row_form(clickhouse) -> None:
    source, recorder = clickhouse

    source.schema()

    assert len(recorder.sql) == 1
    assert "1 = 0" in recorder.sql[0]
    assert "SELECT * FROM big" in recorder.sql[0], "the user's query must still be wrapped"


def test_a_real_read_is_unaffected(clickhouse) -> None:
    """The probe must not leak into the read path — `WHERE 1 = 0` there returns nothing."""
    source, recorder = clickhouse

    rows = sum(b.num_rows for b in source.read())

    assert rows == _BIG
    assert "1 = 0" not in recorder.sql[-1]


def test_schema_then_read_does_not_submit_the_query_twice(clickhouse) -> None:
    """The compounding cost: the planner asks for the schema, then executes."""
    source, recorder = clickhouse

    source.schema()
    source.read()

    full_scans = sum(1 for s in recorder.sql if "1 = 0" not in s)
    assert full_scans == 1, f"the query was submitted in full {full_scans} times"


# ---- the guard: a probe that cannot be trusted must not be believed ------------


def test_an_untyped_probe_falls_back_to_the_full_read(monkeypatch) -> None:
    """A driver that typed an empty result as `null` would silently mistype every column.

    That is the same broken `schema()` contract the CSV and Avro fixes were about, and it
    would have been *introduced* by the optimization. So an untyped probe is rejected and
    the connector pays for the full read rather than reporting a wrong type.
    """
    from batcher.io.formats.sql import clickhouse as mod

    def _untyped_if_probe(sql: str) -> pa.Table:
        if "1 = 0" in sql:  # a driver that infers types from rows it never saw
            return pa.table({"k": pa.array([], pa.null()), "v": pa.array([], pa.null())})
        return _result(sql)

    class Untyped(_Recorder):
        def query_arrow(self, sql: str) -> pa.Table:
            self.sql.append(sql)
            return _untyped_if_probe(sql)

        # `schema()` reads the stream header, so the untyped driver has to be simulated
        # there too. Overriding only `query_arrow` left the probe looking perfectly typed
        # and the fallback never exercised — the test passed while testing nothing.
        def query_arrow_stream(self, sql: str) -> _ArrowStream:
            self.sql.append(sql)
            return _ArrowStream(_untyped_if_probe(sql), lambda rows: None)

    recorder = Untyped()
    monkeypatch.setattr(mod, "_client", lambda params: recorder)

    schema = mod.ClickHouseSource(query="SELECT * FROM big", host="h").schema()

    assert schema == _SCHEMA, "an untyped probe must not be reported as the schema"
    assert any("1 = 0" not in s for s in recorder.sql), "it should have fallen back"


def test_probe_is_typed_accepts_a_real_schema() -> None:
    from batcher.io.formats.sql._common import probe_is_typed

    assert probe_is_typed(_SCHEMA)


def test_probe_is_typed_rejects_null_columns_and_empty_schemas() -> None:
    from batcher.io.formats.sql._common import probe_is_typed

    assert not probe_is_typed(pa.schema([("k", pa.int64()), ("v", pa.null())]))
    assert not probe_is_typed(pa.schema([]))


# ---- the shared SQL shaping ---------------------------------------------------


def test_the_probe_wraps_a_query_without_parsing_it() -> None:
    from batcher.io.formats.sql._common import schema_probe

    sql = schema_probe("SELECT a, b FROM t ORDER BY a")

    assert "1 = 0" in sql
    assert "SELECT a, b FROM t ORDER BY a" in sql


def test_the_probe_handles_a_table_read() -> None:
    """A `table=` source has no query string of its own."""
    from batcher.io.formats.sql._common import schema_probe

    sql = schema_probe(None, table="warehouse.public.events")

    assert "warehouse.public.events" in sql
    assert "1 = 0" in sql
