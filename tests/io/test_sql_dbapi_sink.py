"""The row-level SQL write path, against a real database.

`sqlite3` ships with the standard library and implements ``ON CONFLICT``, so every claim
about the *behavior* of a write here is a genuine database round trip rather than a string
assertion — which matters, because the failures this path has are ones a generated-SQL
assertion cannot see. An upsert against a table with no primary key produces perfectly
reasonable-looking SQL and duplicates every row.

The dialects Batcher cannot run in CI — MySQL's ``ON DUPLICATE KEY``, SQL Server's and
Oracle's ``MERGE`` — are pinned as generated text instead, which is the honest limit: it
proves the statement is the shape the dialect documents, not that a server accepted it.
Those cases say so in their names.

Three defects are pinned here because each one passed every gate while being wrong:

1. ``ds.write.sql(table, uri=...)`` dropped `mode` entirely. The writer's save-mode gate
   consumed it and never forwarded it, so the documented ``mode="replace"`` raised "unknown
   save mode", ``mode="append"`` raised "only supported for delta/iceberg/hudi/snowflake",
   and the default ``mode="overwrite"`` silently *appended*. A save mode that does the
   opposite of what it says is data corruption, not a missing feature.
2. ``mode="error"`` asked the local filesystem whether a file named ``orders`` existed. It
   never did, so the mode whose only job is to refuse an existing destination permitted
   every write.
3. A write to any database with no ADBC driver — MySQL, MariaDB, Oracle, SQL Server —
   raised "no ADBC driver" on a database `bt.read.sql` reads happily.
"""

from __future__ import annotations

import datetime
import decimal
import sqlite3

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.sql.dbapi import DBAPISink
from batcher.io.formats.sql.dbapi import source as _source
from batcher.io.formats.sql.dbapi._bind import null_key_rows, parameter_chunks
from batcher.io.formats.sql.dbapi._ddl import create_table, sql_type
from batcher.io.formats.sql.dbapi._dsn import connect_target, driver_for, installed_driver
from batcher.io.formats.sql.dbapi._statements import (
    delete,
    dialect_for_driver,
    insert,
    truncate,
    update,
    upsert,
    upsert_style,
)
from batcher.io.formats.sql.routing import read_backend, write_backend
from batcher.io.formats.sql.uri import parse_uri

pytestmark = pytest.mark.io


@pytest.fixture
def uri(tmp_path):
    """A SQLAlchemy-style URI addressing a fresh local SQLite database."""
    return f"sqlite:///{tmp_path / 'app.db'}"


def rows(uri_: str, table: str = "orders") -> list[tuple]:
    """Every row of `table`, sorted, read with a plain driver rather than through Batcher."""
    conn = sqlite3.connect(uri_.removeprefix("sqlite:///"))
    try:
        return sorted(conn.execute(f'SELECT * FROM "{table}"').fetchall())
    finally:
        conn.close()


# --- 1. Statement generation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("paramstyle", "expected"),
    [
        ("qmark", 'INSERT INTO "t" ("id", "amt") VALUES (?, ?)'),
        ("format", 'INSERT INTO "t" ("id", "amt") VALUES (%s, %s)'),
        ("numeric", 'INSERT INTO "t" ("id", "amt") VALUES (:1, :2)'),
        ("named", 'INSERT INTO "t" ("id", "amt") VALUES (:p0, :p1)'),
        ("pyformat", 'INSERT INTO "t" ("id", "amt") VALUES (%(p0)s, %(p1)s)'),
    ],
)
def test_every_pep249_paramstyle_is_rendered(paramstyle, expected) -> None:
    stmt = insert("t", ("id", "amt"), dialect="postgresql", paramstyle=paramstyle)
    assert stmt.sql == expected
    assert stmt.positional is (paramstyle in ("qmark", "format", "numeric"))


def test_an_unknown_paramstyle_is_refused_rather_than_guessed() -> None:
    with pytest.raises(BackendError, match="paramstyle"):
        insert("t", ("id",), dialect="postgresql", paramstyle="dollar")


def test_a_schema_qualified_table_is_quoted_part_by_part() -> None:
    stmt = insert("public.orders", ("id",), dialect="postgresql", paramstyle="qmark")
    assert '"public"."orders"' in stmt.sql


@pytest.mark.parametrize(
    ("dialect", "style"),
    [
        ("postgresql", "on_conflict"),
        ("sqlite", "on_conflict"),
        ("duckdb", "on_conflict"),
        ("cockroachdb", "on_conflict"),
        ("mysql", "on_duplicate_key"),
        ("mariadb", "on_duplicate_key"),
        ("mssql", "merge"),
        ("oracle", "merge"),
        ("snowflake", "merge"),
        ("informix", None),
    ],
)
def test_each_dialect_family_maps_to_its_upsert_spelling(dialect, style) -> None:
    assert upsert_style(dialect) == style


def test_generated_upsert_for_mysql_matches_the_documented_syntax() -> None:
    """Pinned as text: no MySQL server runs in CI, so this proves shape, not acceptance."""
    stmt = upsert("t", ("id", "amt"), ("id",), dialect="mysql", paramstyle="format")
    assert stmt.sql == (
        "INSERT INTO `t` (`id`, `amt`) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE `amt` = VALUES(`amt`)"
    )


def test_generated_merge_for_sql_server_matches_the_documented_syntax() -> None:
    """Pinned as text, with SQL Server's mandatory statement terminator."""
    stmt = upsert("t", ("id", "amt"), ("id",), dialect="mssql", paramstyle="qmark")
    assert stmt.sql == (
        "MERGE INTO [t] AS t USING (SELECT ? AS [id], ? AS [amt]) AS s ON (t.[id] = s.[id]) "
        "WHEN MATCHED THEN UPDATE SET t.[amt] = s.[amt] "
        "WHEN NOT MATCHED THEN INSERT ([id], [amt]) VALUES (s.[id], s.[amt]);"
    )


def test_generated_merge_for_oracle_selects_from_dual_and_drops_the_as_alias() -> None:
    stmt = upsert("t", ("id", "amt"), ("id",), dialect="oracle", paramstyle="named")
    assert " FROM dual)" in stmt.sql
    assert " AS t " not in stmt.sql
    assert not stmt.sql.endswith(";")


def test_an_upsert_with_no_updatable_column_does_nothing_rather_than_nothing_valid() -> None:
    stmt = upsert("t", ("id",), ("id",), dialect="postgresql", paramstyle="qmark")
    assert stmt.sql.endswith('ON CONFLICT ("id") DO NOTHING')


def test_an_upsert_into_an_unknown_dialect_names_the_portable_alternative() -> None:
    with pytest.raises(BackendError, match="delete_insert"):
        upsert("t", ("id", "a"), ("id",), dialect="informix", paramstyle="qmark")


def test_update_binds_assigned_columns_before_key_columns() -> None:
    stmt = update("t", ("id", "amt", "note"), ("id",), dialect="sqlite", paramstyle="qmark")
    assert stmt.columns == ("amt", "note", "id")
    assert stmt.sql == 'UPDATE "t" SET "amt" = ?, "note" = ? WHERE "id" = ?'


def test_update_with_nothing_to_set_is_refused() -> None:
    with pytest.raises(BackendError, match="nothing to set"):
        update("t", ("id",), ("id",), dialect="sqlite", paramstyle="qmark")


def test_delete_without_keys_is_refused_because_it_would_empty_the_table() -> None:
    with pytest.raises(BackendError, match="empties the table"):
        delete("t", (), dialect="sqlite", paramstyle="qmark")


def test_a_key_column_absent_from_the_frame_is_named() -> None:
    with pytest.raises(BackendError, match="not columns of the data"):
        upsert("t", ("id",), ("missing",), dialect="sqlite", paramstyle="qmark")


def test_overwrite_uses_delete_not_truncate_so_it_stays_transactional() -> None:
    assert truncate("t", dialect="postgresql") == 'DELETE FROM "t"'


def test_dialect_is_inferred_from_an_unambiguous_driver_only() -> None:
    assert dialect_for_driver("psycopg2") == "postgresql"
    assert dialect_for_driver("sqlite3") == "sqlite"
    assert dialect_for_driver("pyodbc") is None


# --- 2. DDL ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "dialect", "expected"),
    [
        (pa.float64(), "postgresql", "DOUBLE PRECISION"),
        (pa.float64(), "mysql", "DOUBLE"),
        (pa.float64(), "mssql", "FLOAT"),
        (pa.float64(), "oracle", "BINARY_DOUBLE"),
        (pa.float64(), "sqlite", "REAL"),
        (pa.int64(), "postgresql", "BIGINT"),
        (pa.uint64(), "postgresql", "DECIMAL(20, 0)"),
        (pa.decimal128(12, 4), "postgresql", "DECIMAL(12, 4)"),
        (pa.timestamp("us"), "mysql", "DATETIME(6)"),
        (pa.timestamp("us", tz="UTC"), "mssql", "DATETIMEOFFSET"),
        (pa.binary(), "postgresql", "BYTEA"),
        (pa.binary(), "mysql", "BLOB"),
        (pa.dictionary(pa.int32(), pa.string()), "postgresql", "TEXT"),
    ],
)
def test_each_dialect_spells_the_column_type_its_own_way(dtype, dialect, expected) -> None:
    assert sql_type(dtype, dialect) == expected


def test_a_uint64_gets_an_exact_decimal_rather_than_a_signed_column() -> None:
    """A uint64 above 2**63 does not fit BIGINT, and wrapping it is silent corruption."""
    assert "DECIMAL" in sql_type(pa.uint64(), "postgresql")


def test_a_nested_type_is_refused_rather_than_encoded_by_guess() -> None:
    with pytest.raises(BackendError, match="no SQL column type"):
        sql_type(pa.list_(pa.int64()), "postgresql")


def test_created_table_carries_the_primary_key_an_upsert_needs() -> None:
    schema = pa.schema([("id", pa.int64()), ("amt", pa.float64())])
    ddl = create_table("t", schema, dialect="sqlite", key_columns=("id",))
    assert ddl == 'CREATE TABLE "t" ("id" INTEGER NOT NULL, "amt" REAL, PRIMARY KEY ("id"))'


def test_a_key_column_absent_from_the_schema_is_named_before_any_ddl_runs() -> None:
    with pytest.raises(BackendError, match="not columns of the data"):
        create_table("t", pa.schema([("id", pa.int64())]), dialect="sqlite", key_columns=("no",))


# --- 3. Binding -----------------------------------------------------------------------


def test_parameters_are_chunked_so_no_statement_overflows_a_wire_protocol() -> None:
    table = pa.table({"id": list(range(5)), "amt": [float(i) for i in range(5)]})
    stmt = insert("t", ("id", "amt"), dialect="sqlite", paramstyle="qmark")
    chunks = list(parameter_chunks(table, stmt, rows_per_statement=2))
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert chunks[0] == [(0, 0.0), (1, 1.0)]


def test_a_named_paramstyle_binds_mappings_keyed_by_synthetic_names() -> None:
    table = pa.table({"id": [1], "amt": [2.0]})
    stmt = insert("t", ("id", "amt"), dialect="sqlite", paramstyle="named")
    assert list(parameter_chunks(table, stmt, rows_per_statement=10)) == [[{"p0": 1, "p1": 2.0}]]


def test_binding_reorders_to_the_statements_parameter_order_not_the_frames() -> None:
    table = pa.table({"id": [1], "amt": [2.0]})
    stmt = update("t", ("id", "amt"), ("id",), dialect="sqlite", paramstyle="qmark")
    assert list(parameter_chunks(table, stmt, rows_per_statement=10)) == [[(2.0, 1)]]


def test_a_column_the_frame_lacks_is_named_with_what_it_does_have() -> None:
    stmt = insert("t", ("nope",), dialect="sqlite", paramstyle="qmark")
    with pytest.raises(BackendError, match="not in the data being written"):
        list(parameter_chunks(pa.table({"id": [1]}), stmt, rows_per_statement=1))


def test_null_keys_are_counted_because_they_match_no_row_on_any_database() -> None:
    assert null_key_rows(pa.table({"a": [1, None], "b": [None, 2]}), ("a", "b")) == 2
    assert null_key_rows(pa.table({"a": [1, 2]}), ("a",)) == 0


# --- 4. Connection resolution ---------------------------------------------------------


def test_a_file_backed_uri_keeps_the_absolute_relative_distinction() -> None:
    assert connect_target(parse_uri("sqlite:////var/db/a.db"))[1] == {"database": "/var/db/a.db"}
    assert connect_target(parse_uri("sqlite:///a.db"))[1] == {"database": "a.db"}


def test_postgres_drivers_get_dbname_and_everyone_else_gets_database() -> None:
    parsed = parse_uri("postgresql://u:pw@h:5432/app")
    assert "dbname" in connect_target(parsed, module="psycopg2")[1]
    assert "database" in connect_target(parsed, module="pg8000")[1]


def test_query_string_options_reach_the_driver_verbatim() -> None:
    _, kwargs = connect_target(parse_uri("postgresql://h/app?sslmode=require"), module="psycopg")
    assert kwargs["sslmode"] == "require"


def test_oracle_gets_an_easy_connect_dsn_rather_than_host_and_database() -> None:
    _, kwargs = connect_target(parse_uri("oracle://u:p@h:1521/XEPDB1"), module="oracledb")
    assert kwargs == {"user": "u", "password": "p", "dsn": "h:1521/XEPDB1"}


def test_a_password_reference_is_carried_unresolved_so_it_never_enters_a_pickle() -> None:
    _, kwargs = connect_target(
        parse_uri("postgresql://u@h/app", password="env:PGPASSWORD"), module="psycopg"
    )
    assert kwargs["password"] == "env:PGPASSWORD"


def test_a_scheme_with_a_dedicated_connector_names_it_instead_of_guessing_a_dsn() -> None:
    with pytest.raises(BackendError, match=r"write\.snowflake"):
        driver_for("snowflake")


def test_installed_driver_answers_none_rather_than_raising() -> None:
    assert installed_driver("sqlite") == "sqlite3"
    assert installed_driver("nonesuch") is None


# --- 5. Backend routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "opts", "expected"),
    [
        ("upsert", {"uri": "postgresql://h/db"}, "dbapi"),
        ("update", {"uri": "postgresql://h/db"}, "dbapi"),
        ("delete", {"uri": "postgresql://h/db"}, "dbapi"),
        ("append", {"uri": "mysql://h/db"}, "dbapi"),
        ("append", {"module": "sqlite3"}, "dbapi"),
        ("append", {"driver": "adbc_driver_postgresql"}, "adbc"),
        ("append", {}, "adbc"),
    ],
)
def test_the_write_router_picks_the_backend_that_can_serve_the_call(mode, opts, expected) -> None:
    assert write_backend(mode, opts) == expected


def test_the_write_router_falls_back_when_the_arrow_driver_is_not_installed() -> None:
    """`sqlite3` ships with Python; `adbc_driver_sqlite` almost never does."""
    assert write_backend("append", {"uri": "sqlite:///a.db"}) == "dbapi"


def test_the_read_router_honors_an_explicitly_named_pep249_driver() -> None:
    assert read_backend(None, {"module": "psycopg"}) == "dbapi"
    assert read_backend("sqlite:///a.db", {}) == "dbapi"


# --- 6. End to end, against a real database -------------------------------------------


def test_append_creates_the_table_and_writes_every_row(uri) -> None:
    ds = bt.from_pydict({"id": [1, 2, 3], "amt": [10.0, 20.0, 30.0]})
    manifest = ds.write.sql("orders", uri=uri, mode="append", key_columns="id")
    assert manifest.total_rows == 3
    assert rows(uri) == [(1, 10.0), (2, 20.0), (3, 30.0)]


def test_upsert_updates_the_matching_key_and_inserts_the_rest(uri) -> None:
    bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2, 3], "amt": [99.0, 3.0]}).write.sql(
        "orders", uri=uri, mode="upsert", key_columns="id"
    )
    assert rows(uri) == [(1, 1.0), (2, 99.0), (3, 3.0)]


def test_upsert_is_idempotent_which_is_what_makes_a_retry_safe(uri) -> None:
    change = bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]})
    for _ in range(3):
        change.write.sql("orders", uri=uri, mode="upsert", key_columns="id")
    assert rows(uri) == [(1, 1.0), (2, 2.0)]


def test_upsert_on_a_composite_key(uri) -> None:
    bt.from_pydict({"a": [1, 1], "b": ["x", "y"], "v": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="upsert", key_columns=["a", "b"]
    )
    bt.from_pydict({"a": [1], "b": ["y"], "v": [9.0]}).write.sql(
        "orders", uri=uri, mode="upsert", key_columns=["a", "b"]
    )
    assert rows(uri) == [(1, "x", 1.0), (1, "y", 9.0)]


def test_update_changes_matching_rows_and_inserts_nothing(uri) -> None:
    bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2, 99], "amt": [22.0, 0.0]}).write.sql(
        "orders", uri=uri, mode="update", key_columns="id"
    )
    assert rows(uri) == [(1, 1.0), (2, 22.0)]


def test_delete_removes_only_the_named_keys(uri) -> None:
    bt.from_pydict({"id": [1, 2, 3], "amt": [1.0, 2.0, 3.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2]}).write.sql("orders", uri=uri, mode="delete", key_columns="id")
    assert rows(uri) == [(1, 1.0), (3, 3.0)]


def test_delete_insert_is_an_upsert_built_from_ansi_sql_alone(uri) -> None:
    bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2, 3], "amt": [22.0, 3.0]}).write.sql(
        "orders", uri=uri, mode="delete_insert", key_columns="id"
    )
    assert rows(uri) == [(1, 1.0), (2, 22.0), (3, 3.0)]


def test_overwrite_replaces_the_tables_contents(uri) -> None:
    bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [7], "amt": [7.0]}).write.sql(
        "orders", uri=uri, mode="overwrite", key_columns="id"
    )
    assert rows(uri) == [(7, 7.0)]


def test_an_empty_overwrite_empties_the_table_and_an_empty_upsert_does_not(uri) -> None:
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    empty = bt.from_pydict({"id": [1], "amt": [1.0]}).filter(bt.col("id") < 0)
    empty.write.sql("orders", uri=uri, mode="upsert", key_columns="id")
    assert rows(uri) == [(1, 1.0)]
    empty.write.sql("orders", uri=uri, mode="overwrite", key_columns="id")
    assert rows(uri) == []


def test_a_write_and_the_read_that_follows_it_are_spelled_the_same_way(uri) -> None:
    bt.from_pydict({"id": [1, 2], "amt": [1.0, 2.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    back = bt.read.sql("SELECT * FROM orders", uri=uri).sort("id")
    assert back.to_pydict() == {"id": [1, 2], "amt": [1.0, 2.0]}


def test_more_rows_than_one_statement_binds_still_all_arrive(uri) -> None:
    n = 250
    bt.from_pydict({"id": list(range(n)), "amt": [float(i) for i in range(n)]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id", rows_per_statement=7
    )
    assert len(rows(uri)) == n


# --- 7. Type fidelity through the sink and back ---------------------------------------

#: Arrow type -> the values written, and what SQLite gives back.
#:
#: SQLite is deliberately the harshest reasonable oracle for this: its column types are
#: *affinities* rather than types, so anything it cannot store natively comes back as
#: whatever the driver adapted it to. Pinning that is the point — a `date32` returning as
#: text still joins, still sorts and still counts, so nothing else would notice.
_FIDELITY: dict[str, tuple[pa.Array, list]] = {
    "int64": (pa.array([1, -2, 2**62], pa.int64()), [-2, 1, 2**62]),
    "float64": (pa.array([1.5, -0.0, 2.0], pa.float64()), [-0.0, 1.5, 2.0]),
    "string": (pa.array(["a", "", "ünï"], pa.string()), ["", "a", "ünï"]),
    "bool": (pa.array([True, False, True], pa.bool_()), [0, 1, 1]),
    "binary": (pa.array([b"\x00\x01", b"", b"z"], pa.binary()), [b"", b"\x00\x01", b"z"]),
}


@pytest.mark.parametrize("type_name", sorted(_FIDELITY))
def test_values_survive_the_write_intact(uri, type_name) -> None:
    values, expected = _FIDELITY[type_name]
    table = pa.table({"id": pa.array(range(len(values)), pa.int64()), "v": values})
    DBAPISink(uri=uri, mode="append", key_columns=("id",)).write(table, "t")
    got = sorted(r[1] for r in rows(uri, "t"))
    assert got == expected


def test_null_values_survive_as_nulls(uri) -> None:
    table = pa.table({"id": [1, 2], "v": pa.array([None, 5], pa.int64())})
    DBAPISink(uri=uri, mode="append", key_columns=("id",)).write(table, "t")
    assert rows(uri, "t") == [(1, None), (2, 5)]


def test_a_date_reaches_the_driver_as_a_python_date(uri) -> None:
    """Not a fidelity claim about SQLite — a claim that nothing is dropped on the way."""
    table = pa.table(
        {"id": pa.array([1], pa.int64()), "d": pa.array([datetime.date(2024, 1, 2)], pa.date32())}
    )
    DBAPISink(uri=uri, mode="append", key_columns=("id",)).write(table, "t")
    assert len(rows(uri, "t")) == 1


def test_a_type_the_driver_will_not_bind_names_the_columns_and_the_fix(uri) -> None:
    """SQLite has no decimal type and its driver refuses to adapt one.

    The driver says "Error binding parameter 3", which is meaningless to someone holding a
    frame. Batcher cannot make the write work without choosing a lossy encoding on the
    user's behalf, so it says which columns were being bound and how to cast instead.
    """
    table = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "n": pa.array([decimal.Decimal("1.25")], pa.decimal128(6, 2)),
        }
    )
    with pytest.raises(BackendError, match="would not bind"):
        DBAPISink(uri=uri, mode="append", key_columns=("id",)).write(table, "t")


# --- 8. Transactions, retries and refusals --------------------------------------------


def test_a_failed_write_leaves_no_partial_batch_behind(uri, monkeypatch) -> None:
    """The whole point of one transaction per write: a half-applied batch is never visible."""
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    calls = {"n": 0}
    original = DBAPISink._execute

    def explode(self, cursor, sql, chunk, path, columns=()):
        calls["n"] += 1
        if calls["n"] > 1:
            raise sqlite3.OperationalError("no such column: boom")
        return original(self, cursor, sql, chunk, path, columns)

    monkeypatch.setattr(DBAPISink, "_execute", explode)
    table = pa.table({"id": [10, 11, 12, 13], "amt": [1.0, 2.0, 3.0, 4.0]})
    sink = DBAPISink(uri=uri, mode="append", key_columns=("id",), retries=0, rows_per_statement=1)
    with pytest.raises(sqlite3.OperationalError):
        sink.write(table, "orders")
    assert rows(uri) == [(1, 1.0)], "a rolled-back write left rows behind"


def test_a_transient_failure_is_retried_and_a_permanent_one_is_not(uri, monkeypatch) -> None:
    attempts = {"n": 0}
    original = DBAPISink._apply

    def flaky(self, table, path):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return original(self, table, path)

    monkeypatch.setattr(DBAPISink, "_apply", flaky)
    sink = DBAPISink(uri=uri, mode="append", key_columns=("id",), retries=3, retry_backoff_s=0)
    sink.write(pa.table({"id": [1], "amt": [1.0]}), "orders")
    assert attempts["n"] == 3
    assert rows(uri) == [(1, 1.0)]


def test_a_permanent_failure_is_raised_on_the_first_attempt(uri, monkeypatch) -> None:
    attempts = {"n": 0}

    def broken(self, table, path):
        attempts["n"] += 1
        raise sqlite3.OperationalError("no such table: nope")

    monkeypatch.setattr(DBAPISink, "_apply", broken)
    sink = DBAPISink(uri=uri, mode="append", retries=5, retry_backoff_s=0)
    with pytest.raises(sqlite3.OperationalError):
        sink.write(pa.table({"id": [1]}), "orders")
    assert attempts["n"] == 1


def test_a_borrowed_connection_is_left_for_its_owner_to_commit(tmp_path) -> None:
    path = tmp_path / "app.db"
    owner = sqlite3.connect(path)
    owner.execute('CREATE TABLE "orders" ("id" INTEGER PRIMARY KEY, "amt" REAL)')
    owner.commit()
    DBAPISink(connection=owner, dialect="sqlite", mode="append", key_columns=("id",)).write(
        pa.table({"id": [1], "amt": [1.0]}), "orders"
    )
    observer = sqlite3.connect(path)
    assert observer.execute("SELECT count(*) FROM orders").fetchone()[0] == 0
    owner.commit()
    assert observer.execute("SELECT count(*) FROM orders").fetchone()[0] == 1
    owner.close()
    observer.close()


class _AbortingConnection:
    """A connection that aborts its transaction on any failed statement, as PostgreSQL does.

    Every statement after a failure raises until `rollback`, which is exactly the behavior
    that makes "probe for the table, then create it" fail on PostgreSQL with an error naming
    neither the table nor the probe.
    """

    def __init__(self) -> None:
        self.aborted = False
        self.statements: list[str] = []
        self.rollbacks = 0

    class _Cursor:
        def __init__(self, conn) -> None:
            self._conn = conn

        def execute(self, sql, *args):
            if self._conn.aborted:
                raise RuntimeError("current transaction is aborted, commands ignored")
            self._conn.statements.append(sql)
            if sql.startswith("SELECT"):
                self._conn.aborted = True
                raise RuntimeError('relation "orders" does not exist')

        def executemany(self, sql, rows):
            self.execute(sql)

        def fetchall(self):
            return []

        def close(self) -> None:
            pass

        rowcount = -1

    def cursor(self):
        return self._Cursor(self)

    def rollback(self) -> None:
        self.rollbacks += 1
        self.aborted = False

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_a_failed_existence_probe_is_rolled_back_before_the_create(monkeypatch) -> None:
    """Without the rollback the CREATE fails on PostgreSQL, on every first write."""
    conn = _AbortingConnection()
    monkeypatch.setattr(_source, "_connect", lambda *a, **k: conn)
    sink = DBAPISink(
        module="psycopg",
        connect_kwargs={},
        dialect="postgresql",
        paramstyle="pyformat",
        mode="append",
        key_columns=("id",),
        retries=0,
    )
    sink.write(pa.table({"id": [1], "amt": [1.0]}), "orders")
    assert conn.rollbacks == 1, "the aborted probe was not unwound"
    assert any(s.startswith("CREATE TABLE") for s in conn.statements)
    assert any(s.startswith("INSERT INTO") for s in conn.statements)


def test_a_borrowed_connection_is_never_probed(tmp_path) -> None:
    """Probing a missing table would abort a transaction Batcher does not own."""
    path = tmp_path / "app.db"
    owner = sqlite3.connect(path)
    sink = DBAPISink(connection=owner, dialect="sqlite", mode="append", key_columns=("id",))
    with pytest.raises(BackendError, match="no such table"):
        sink.write(pa.table({"id": [1], "amt": [1.0]}), "orders")
    owner.close()


def test_a_borrowed_connection_writes_into_a_table_that_exists(tmp_path) -> None:
    path = tmp_path / "app.db"
    owner = sqlite3.connect(path)
    owner.execute('CREATE TABLE "orders" ("id" INTEGER PRIMARY KEY, "amt" REAL)')
    DBAPISink(connection=owner, dialect="sqlite", mode="upsert", key_columns=("id",)).write(
        pa.table({"id": [1], "amt": [1.0]}), "orders"
    )
    owner.commit()
    assert owner.execute("SELECT count(*) FROM orders").fetchone()[0] == 1
    owner.close()


def test_a_borrowed_connection_is_committed_when_the_caller_asks(tmp_path) -> None:
    path = tmp_path / "app.db"
    owner = sqlite3.connect(path)
    owner.execute('CREATE TABLE "orders" ("id" INTEGER PRIMARY KEY, "amt" REAL)')
    owner.commit()
    DBAPISink(
        connection=owner,
        dialect="sqlite",
        mode="append",
        key_columns=("id",),
        commit_writes=True,
    ).write(pa.table({"id": [1], "amt": [1.0]}), "orders")
    observer = sqlite3.connect(path)
    assert observer.execute("SELECT count(*) FROM orders").fetchone()[0] == 1
    owner.close()
    observer.close()


def test_a_destructive_mode_is_refused_past_the_first_shard(uri) -> None:
    sink = DBAPISink(uri=uri, mode="overwrite")
    with pytest.raises(BackendError, match="distributed write"):
        sink.write_partitioned(pa.table({"id": [1]}), "orders", file_index=1)


@pytest.mark.parametrize("mode", ["append", "upsert", "update", "delete"])
def test_the_key_scoped_modes_are_allowed_across_shards(uri, mode) -> None:
    """A shard only ever touches the keys its own rows name, so shards cannot collide."""
    bt.from_pydict({"id": [1], "amt": [0.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    # `append` must not collide with the seeded key; the keyed modes must find it.
    key = 2 if mode == "append" else 1
    sink = DBAPISink(uri=uri, mode=mode, key_columns=("id",))
    written = sink.write_partitioned(pa.table({"id": [key], "amt": [1.0]}), "orders", file_index=2)
    assert written[0].rows == 1


def test_a_keyed_mode_with_no_key_columns_is_refused_at_construction(uri) -> None:
    with pytest.raises(BackendError, match="needs key_columns"):
        DBAPISink(uri=uri, mode="upsert")


def test_an_unknown_mode_lists_the_ones_that_exist(uri) -> None:
    with pytest.raises(BackendError, match="unknown SQL write mode"):
        DBAPISink(uri=uri, mode="merge_into")


def test_an_upsert_against_a_keyless_table_says_why_it_cannot_work(uri) -> None:
    """The failure whose cause is invisible from the driver's own message."""
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql("orders", uri=uri, mode="append")
    with pytest.raises(BackendError, match="PRIMARY KEY or UNIQUE constraint"):
        bt.from_pydict({"id": [1], "amt": [2.0]}).write.sql(
            "orders", uri=uri, mode="upsert", key_columns="id"
        )


def test_null_keys_are_reported_rather_than_silently_matching_nothing(uri, caplog) -> None:
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    with caplog.at_level("WARNING"):
        DBAPISink(uri=uri, mode="update", key_columns=("id",)).write(
            pa.table({"id": pa.array([None], pa.int64()), "amt": [9.0]}), "orders"
        )
    assert any("null key" in record.getMessage() for record in caplog.records)


# --- 9. The save-mode contract the writer used to drop --------------------------------


def test_the_save_mode_reaches_the_sink_rather_than_being_consumed_by_the_gate(uri) -> None:
    """`mode="overwrite"` used to be swallowed, so the write silently appended."""
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2], "amt": [2.0]}).write.sql(
        "orders", uri=uri, mode="overwrite", key_columns="id"
    )
    assert rows(uri) == [(2, 2.0)]


def test_append_is_no_longer_refused_for_a_database_sink(uri) -> None:
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql("orders", uri=uri, mode="append")
    bt.from_pydict({"id": [2], "amt": [2.0]}).write.sql("orders", uri=uri, mode="append")
    assert rows(uri) == [(1, 1.0), (2, 2.0)]


def test_an_unknown_mode_is_still_rejected_by_the_save_mode_gate(uri) -> None:
    with pytest.raises((PlanError, BackendError)):
        bt.from_pydict({"id": [1]}).write.sql("orders", uri=uri, mode="nonsense")


def test_the_adbc_sink_translates_a_save_mode_to_an_ingest_disposition() -> None:
    from batcher.io.formats.sql.adbc.sink import ADBCSink

    assert ADBCSink(uri="postgresql://h/db", mode="overwrite").mode == "replace"
    assert ADBCSink(uri="postgresql://h/db", mode="append").mode == "create_append"
    assert ADBCSink(uri="postgresql://h/db", mode="create_append").mode == "create_append"
    with pytest.raises(BackendError, match="unknown ADBC write mode"):
        ADBCSink(uri="postgresql://h/db", mode="nonsense")


def test_partition_by_is_refused_rather_than_silently_dropped(uri) -> None:
    """A table has no Hive layout, so honoring it is impossible and ignoring it is silent."""
    with pytest.raises(PlanError, match="no meaning"):
        bt.from_pydict({"id": [1], "amt": [1.0]}).write(
            "orders", "dbapi", uri=uri, partition_by=["id"]
        )


def test_a_standing_repartition_layout_does_not_trip_that_refusal(uri) -> None:
    """`repartition(by=...)` is a layout hint, not a request this write made."""
    bt.from_pydict({"id": [1], "amt": [1.0]}).repartition(by="id").write(
        "orders", "dbapi", uri=uri, key_columns=("id",)
    )
    assert rows(uri) == [(1, 1.0)]


def test_the_dbapi_sink_is_registered() -> None:
    from batcher.io.formats import SINKS

    assert SINKS.get("dbapi") is DBAPISink


# --- 10. What the DB-API read can now push --------------------------------------------
#
# This path materializes every value as a Python object, so anything the server can be made
# to do instead is worth more here than on any Arrow-native connector. Until the source
# could name its dialect it pushed only the predicate: no row cap, no top-N, and no
# identifier quoting at all.


@pytest.fixture
def orders(tmp_path):
    """A hundred-row table whose first column is a SQL reserved word."""
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE "orders" ("order" INTEGER, "amount" REAL)')
    conn.executemany('INSERT INTO "orders" VALUES (?, ?)', [(i, float(i)) for i in range(100)])
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


def test_the_dialect_comes_from_the_uri_and_then_from_the_driver() -> None:
    from batcher.io.formats.sql.dbapi import DBAPISource

    assert DBAPISource(uri="sqlite:///a.db", table="t").dialect == "sqlite"
    assert DBAPISource(module="psycopg2", table="t").dialect == "postgresql"
    assert DBAPISource(module="pyodbc", table="t").dialect is None


def test_a_row_cap_is_pushed_into_the_sql(orders) -> None:
    from batcher.io.formats.sql.dbapi import DBAPISource

    source = DBAPISource(uri=orders, table="orders")
    assert source.supports_limit
    assert source._pushed(limit=3).rstrip().endswith("LIMIT 3")
    assert sum(batch.num_rows for batch in source.read(limit=3)) == 3


def test_a_top_n_carries_its_ordering_and_its_nulls_clause(orders) -> None:
    from batcher.io.formats.sql.dbapi import DBAPISource

    pushed = DBAPISource(uri=orders, table="orders")._pushed(
        limit=2, ordering=(("amount", True, False),)
    )
    assert "ORDER BY" in pushed and "NULLS LAST" in pushed and pushed.endswith("LIMIT 2")


def test_a_cap_is_dropped_when_its_ordering_cannot_be_pushed(orders) -> None:
    """The first n of an unordered read is not the first n of a sorted one."""
    from batcher.io.formats.sql.dbapi import DBAPISource

    source = DBAPISource(uri=orders, table="orders", dialect="mysql")
    assert not source.supports_ordering
    assert "LIMIT" not in source._pushed(limit=2, ordering=(("amount", True, False),))


def test_an_unknown_dialect_pushes_no_cap_at_all(orders) -> None:
    """A cap the server cannot parse turns a working query into a syntax error."""
    from batcher.io.formats.sql.dbapi import DBAPISource

    source = DBAPISource(module="pyodbc", connect_kwargs={}, table="t")
    assert not source.supports_limit
    assert "LIMIT" not in source._pushed(limit=5)


def test_a_reserved_word_column_is_quoted_so_the_projection_parses(orders) -> None:
    """`SELECT order FROM ...` is a syntax error, not a slow query."""
    back = bt.read.sql(None, uri=orders, table="orders").select("order").head(2)
    assert back.to_pydict() == {"order": [0, 1]}


def test_head_reads_only_what_it_asked_for(orders) -> None:
    assert bt.read.sql("SELECT * FROM orders", uri=orders).head(3).to_pydict() == {
        "order": [0, 1, 2],
        "amount": [0.0, 1.0, 2.0],
    }


def test_a_point_lookup_still_returns_the_right_row(orders) -> None:
    found = bt.read.sql(None, uri=orders, table="orders").filter(bt.col("order") == 7)
    assert found.to_pydict() == {"order": [7], "amount": [7.0]}


# --- 11. Streaming into an operational table ------------------------------------------


def _stream_source(tmp_path):
    """An incremental-file stream over two rows of newline-delimited JSON."""
    import json

    source = tmp_path / "incoming"
    source.mkdir(parents=True)
    with (source / "a.json").open("w") as handle:
        for row in ({"id": 1, "amount": 1.0}, {"id": 2, "amount": 2.0}):
            handle.write(json.dumps(row) + "\n")
    return bt.read.files_incremental(str(source), "json", state_dir=str(tmp_path / "state"))


def _drain(query) -> None:
    for method in ("await_termination", "awaitTermination", "wait"):
        if hasattr(query, method):
            getattr(query, method)()
            return


def test_a_streaming_upsert_reaches_the_table(tmp_path, uri) -> None:
    """The canonical streaming-OLTP shape: micro-batches maintained onto a keyed table."""
    _drain(
        _stream_source(tmp_path).write(
            "orders",
            "dbapi",
            uri=uri,
            mode="upsert",
            key_columns=("id",),
            trigger=bt.Trigger.available_now(),
            query_name="orders-upsert",
        )
    )
    assert rows(uri) == [(1, 1.0), (2, 2.0)]


def test_the_dml_mode_is_not_swallowed_on_the_way_to_the_stream_sink(tmp_path, uri) -> None:
    """Dropped, `mode="upsert"` became an append and every replayed batch duplicated a row."""
    for run in range(2):
        _drain(
            _stream_source(tmp_path / f"run{run}").write(
                "orders",
                "dbapi",
                uri=uri,
                mode="upsert",
                key_columns=("id",),
                trigger=bt.Trigger.available_now(),
                query_name="orders-upsert",
            )
        )
    assert rows(uri) == [(1, 1.0), (2, 2.0)], "a replayed micro-batch duplicated rows"


def test_a_keyed_stream_write_is_not_warned_about_as_at_least_once(tmp_path, uri) -> None:
    """An upsert absorbs a replay without any transaction log; saying otherwise is wrong."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _drain(
            _stream_source(tmp_path).write(
                "orders",
                "dbapi",
                uri=uri,
                mode="upsert",
                key_columns=("id",),
                trigger=bt.Trigger.available_now(),
            )
        )
    assert not [w for w in caught if "at-least-once" in str(w.message)]


def test_a_streaming_append_is_still_warned_about(tmp_path, uri) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _drain(
            _stream_source(tmp_path).write(
                "orders",
                "dbapi",
                uri=uri,
                mode="append",
                trigger=bt.Trigger.available_now(),
            )
        )
    assert [w for w in caught if "at-least-once" in str(w.message)]


# --- 12. The Arrow-native path gained the same three pushdowns ------------------------


def test_the_adbc_source_names_its_dialect_from_the_uri_or_the_driver() -> None:
    from batcher.io.formats.sql.adbc import ADBCSource

    assert ADBCSource(query="q", uri="postgresql://h/db").dialect == "postgresql"
    assert ADBCSource(query="q", driver="adbc_driver_sqlite", db_kwargs={}).dialect == "sqlite"
    assert ADBCSource(query="q", driver="something_custom", db_kwargs={}).dialect is None


def test_the_adbc_source_quotes_identifiers_and_pushes_a_cap() -> None:
    from batcher.io.formats.sql.adbc import ADBCSource

    source = ADBCSource(query="SELECT * FROM t", uri="postgresql://h/db")
    pushed = source._sql(projection=["order"], limit=5)
    assert 'SELECT "order"' in pushed
    assert pushed.rstrip().endswith("LIMIT 5")


def test_an_adbc_driver_with_no_known_dialect_changes_nothing() -> None:
    """Quoting with the wrong delimiter is a new failure; not quoting is what it always did."""
    from batcher.io.formats.sql.adbc import ADBCSource

    source = ADBCSource(query="SELECT * FROM t", driver="something_custom", db_kwargs={})
    assert not source.supports_limit
    pushed = source._sql(projection=["order"], limit=5)
    assert "SELECT order" in pushed and "LIMIT" not in pushed


def test_a_partition_fragment_quotes_its_key_column() -> None:
    """A partition column is chosen for being indexed, which says nothing about its name."""
    from batcher.io.formats.sql.partition import range_predicates
    from batcher.io.formats.sql.uri import quote_identifier

    fragments = range_predicates(
        "order", 0, 100, 2, quote=lambda name: quote_identifier(name, "postgresql")
    )
    assert fragments[1] == '"order" >= 50.0'
    assert range_predicates("id", 0, 100, 2)[1] == "id >= 50.0"


# --- 13. The semantics that decide right from wrong -----------------------------------


def test_an_upsert_of_a_subset_of_columns_merges_rather_than_replaces(uri) -> None:
    """`ON CONFLICT DO UPDATE SET` touches the named columns only; the rest survive.

    That is the opposite of a document store's upsert, which replaces the document, so it
    is worth pinning rather than leaving to be rediscovered.
    """
    bt.from_pydict({"id": [1], "amt": [1.0], "note": ["keep"]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [1], "amt": [9.0]}).write.sql(
        "orders", uri=uri, mode="upsert", key_columns="id"
    )
    assert rows(uri) == [(1, 9.0, "keep")]


def test_a_repeated_key_in_one_upsert_keeps_the_last_row(uri) -> None:
    """One statement per row, so a repeated key collapses in frame order."""
    bt.from_pydict({"id": [1, 1, 2], "amt": [10.0, 20.0, 30.0]}).write.sql(
        "orders", uri=uri, mode="upsert", key_columns="id"
    )
    assert rows(uri) == [(1, 20.0), (2, 30.0)]


def test_a_repeated_key_in_a_delete_insert_says_why_it_failed(uri) -> None:
    """The target reports only that a constraint failed, which names neither cause nor fix."""
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    with pytest.raises(BackendError, match="more than one row per key"):
        bt.from_pydict({"id": [1, 1], "amt": [10.0, 20.0]}).write.sql(
            "orders", uri=uri, mode="delete_insert", key_columns="id"
        )


def test_a_column_the_frame_lacks_takes_the_database_default_on_append(uri) -> None:
    bt.from_pydict({"id": [1], "amt": [1.0]}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    bt.from_pydict({"id": [2]}).write.sql("orders", uri=uri, mode="append", key_columns="id")
    assert rows(uri) == [(1, 1.0), (2, None)]
