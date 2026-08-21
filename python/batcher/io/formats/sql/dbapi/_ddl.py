"""Arrow schema → ``CREATE TABLE`` — the one place a column type is chosen for a database.

A write to a table that does not exist yet has to name a SQL type for every column, and
there is no portable answer: PostgreSQL's ``DOUBLE PRECISION`` is MySQL's ``DOUBLE``, SQL
Server's ``FLOAT`` and Oracle's ``BINARY_DOUBLE``. This module holds that table and
nothing else, so the mapping is one grep away rather than spread through the sink.

## What it refuses

A nested Arrow type — list, struct, map, union — has no portable column type, and picking
one (``JSON``? ``TEXT``? a side table?) is a schema design decision the writer is not
entitled to make on the user's behalf. Those raise, naming the column, so the user creates
the table themselves and writes into it with ``mode="append"``. Every scalar type Batcher
can hold is mapped.

## Why the sink probes instead of writing ``IF NOT EXISTS``

``CREATE TABLE IF NOT EXISTS`` is not portable either — SQL Server and Oracle have no such
clause. Rather than fork the DDL a second time, `DBAPISink` asks whether the table exists
with the same zero-row ``WHERE 1 = 0`` probe every source in this package uses for schema
inference, and skips the ``CREATE`` when it answers. That is dialect-free by construction.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi._statements import qualified_table, quote

__all__ = ["create_table", "sql_type"]

#: The ANSI-ish baseline, applied to any dialect with no override below.
_ANSI: dict[str, str] = {
    "bool": "BOOLEAN",
    "int8": "SMALLINT",
    "int16": "SMALLINT",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "uint8": "SMALLINT",
    "uint16": "INTEGER",
    "uint32": "BIGINT",
    # A uint64 above 2^63 does not fit any signed integer column, and silently wrapping it
    # is the kind of corruption that surfaces months later. An exact decimal holds all 20
    # digits.
    "uint64": "DECIMAL(20, 0)",
    "float16": "REAL",
    "float32": "REAL",
    "float64": "DOUBLE PRECISION",
    "string": "TEXT",
    "binary": "BYTEA",
    "date": "DATE",
    "time": "TIME",
    "timestamp": "TIMESTAMP",
    "timestamp_tz": "TIMESTAMP WITH TIME ZONE",
}

#: Per-dialect deviations from `_ANSI`, keyed by the same logical type names.
_OVERRIDES: dict[str, dict[str, str]] = {
    "mysql": {
        "float64": "DOUBLE",
        "float32": "FLOAT",
        "float16": "FLOAT",
        "binary": "BLOB",
        # MySQL's TIMESTAMP is converted to and from the session time zone and spans only
        # 1970-2038; DATETIME is the naive, full-range type a `timestamp` column means.
        "timestamp": "DATETIME(6)",
        "timestamp_tz": "TIMESTAMP(6)",
        "time": "TIME(6)",
        # TEXT rather than VARCHAR(n) because no row width is known here. The
        # consequence is real and worth stating: MySQL cannot index a TEXT column
        # without a prefix length, so a *string* primary key fails to create and the
        # table has to be created by hand. An integer or date key is unaffected.
        "string": "TEXT",
    },
    "mssql": {
        "bool": "BIT",
        "float64": "FLOAT",
        "float32": "REAL",
        "float16": "REAL",
        "string": "NVARCHAR(MAX)",
        "binary": "VARBINARY(MAX)",
        "timestamp": "DATETIME2",
        "timestamp_tz": "DATETIMEOFFSET",
    },
    "oracle": {
        # Oracle gained a native BOOLEAN only in 23c; NUMBER(1) is what every earlier
        # release stores one as, and 23c accepts it unchanged.
        "bool": "NUMBER(1)",
        "int8": "NUMBER(3)",
        "int16": "NUMBER(5)",
        "int32": "NUMBER(10)",
        "int64": "NUMBER(19)",
        "uint8": "NUMBER(3)",
        "uint16": "NUMBER(5)",
        "uint32": "NUMBER(10)",
        "uint64": "NUMBER(20)",
        "float32": "BINARY_FLOAT",
        "float16": "BINARY_FLOAT",
        "float64": "BINARY_DOUBLE",
        "string": "VARCHAR2(4000)",
        "binary": "BLOB",
        # Oracle has no TIME type; a time of day is stored as an interval there.
        "time": "INTERVAL DAY(0) TO SECOND(6)",
    },
    "sqlite": {
        # SQLite's declared types are advisory (type *affinity*), so the point of naming
        # them is that a later reader and any other tool see the intended type.
        "bool": "INTEGER",
        "int8": "INTEGER",
        "int16": "INTEGER",
        "int32": "INTEGER",
        "int64": "INTEGER",
        "uint8": "INTEGER",
        "uint16": "INTEGER",
        "uint32": "INTEGER",
        "uint64": "INTEGER",
        "float16": "REAL",
        "float32": "REAL",
        "float64": "REAL",
        "binary": "BLOB",
    },
    "bigquery": {
        "int8": "INT64",
        "int16": "INT64",
        "int32": "INT64",
        "int64": "INT64",
        "uint8": "INT64",
        "uint16": "INT64",
        "uint32": "INT64",
        "uint64": "NUMERIC",
        "float16": "FLOAT64",
        "float32": "FLOAT64",
        "float64": "FLOAT64",
        "string": "STRING",
        "binary": "BYTES",
        "timestamp": "DATETIME",
        "timestamp_tz": "TIMESTAMP",
    },
    "clickhouse": {
        "bool": "UInt8",
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "uint8": "UInt8",
        "uint16": "UInt16",
        "uint32": "UInt32",
        "uint64": "UInt64",
        "float16": "Float32",
        "float32": "Float32",
        "float64": "Float64",
        "string": "String",
        "binary": "String",
        "date": "Date",
        "time": "String",
        "timestamp": "DateTime64(6)",
        "timestamp_tz": "DateTime64(6, 'UTC')",
    },
}

#: Dialect families that share one override table.
_FAMILIES: dict[str, str] = {
    "mariadb": "mysql",
    "tidb": "mysql",
    "singlestore": "mysql",
    "memsql": "mysql",
    "percona": "mysql",
    "starrocks": "mysql",
    "doris": "mysql",
    "sqlserver": "mssql",
    "duckdb": "sqlite",
}


def _logical(dtype: pa.DataType) -> str | None:
    """The logical name `_ANSI` and `_OVERRIDES` are keyed by, or None when unmappable."""
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_decimal(dtype):
        return "decimal"
    if pa.types.is_integer(dtype):
        return str(dtype)
    if pa.types.is_floating(dtype):
        # Arrow spells these `halffloat`/`float`/`double`; the table is keyed by bit width
        # so the SQL names read as the sizes they are.
        return {16: "float16", 32: "float32", 64: "float64"}[dtype.bit_width]
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if (
        pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_fixed_size_binary(dtype)
    ):
        return "binary"
    if pa.types.is_date(dtype):
        return "date"
    if pa.types.is_time(dtype):
        return "time"
    if pa.types.is_timestamp(dtype):
        return "timestamp_tz" if dtype.tz else "timestamp"
    if pa.types.is_dictionary(dtype):
        return _logical(dtype.value_type)
    return None


def sql_type(dtype: pa.DataType, dialect: str | None) -> str:
    """The column type `dialect` spells `dtype` as.

    Args:
        dtype: The Arrow type of the column being created.
        dialect: The dialect name, or None for the ANSI baseline.

    Returns:
        A SQL column type.

    Raises:
        BackendError: If `dtype` is nested, or otherwise has no portable column type.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._ddl import sql_type
            >>> sql_type(pa.float64(), "postgresql")
            'DOUBLE PRECISION'
            >>> sql_type(pa.float64(), "mysql")
            'DOUBLE'
    """
    logical = _logical(dtype)
    if logical is None:
        raise BackendError(
            f"no SQL column type for Arrow type {dtype}. Nested and extension types have "
            "no portable spelling, so Batcher will not invent one: create the table "
            "yourself with the encoding you want (JSON, a side table, …) and write into "
            "it with mode='append', or flatten the column first."
        )
    if logical == "decimal":
        return f"DECIMAL({dtype.precision}, {dtype.scale})"
    family = _FAMILIES.get(_normalize(dialect), _normalize(dialect))
    override = _OVERRIDES.get(family, {})
    return override.get(logical) or _ANSI[logical]


def _normalize(dialect: str | None) -> str:
    """Fold a dialect name to the key `_FAMILIES` and `_OVERRIDES` use."""
    return "" if dialect is None else dialect.split("+")[0].strip().lower()


def create_table(
    table: str,
    schema: pa.Schema,
    *,
    dialect: str | None,
    key_columns: tuple[str, ...] = (),
) -> str:
    """``CREATE TABLE`` matching `schema`, with `key_columns` as the primary key.

    The primary key is not decoration: it is what makes an ``ON CONFLICT`` or ``ON
    DUPLICATE KEY`` upsert able to detect a conflict at all. Creating the table without
    one turns a later upsert into an append that silently duplicates every key.

    A key column is emitted ``NOT NULL`` because a primary key requires it, and every
    other column stays nullable — a write that has seen no null yet does not prove the
    column never holds one.

    Args:
        table: The table to create, optionally schema-qualified.
        schema: The Arrow schema of the data being written.
        dialect: The dialect whose type names and identifier quoting apply.
        key_columns: The columns forming the primary key; empty for none.

    Returns:
        The ``CREATE TABLE`` statement text.

    Raises:
        BackendError: If a column's type has no portable SQL spelling, or `key_columns`
            names a column absent from `schema`.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._ddl import create_table
            >>> schema = pa.schema([("id", pa.int64()), ("amt", pa.float64())])
            >>> create_table("t", schema, dialect="sqlite", key_columns=("id",))
            'CREATE TABLE "t" ("id" INTEGER NOT NULL, "amt" REAL, PRIMARY KEY ("id"))'
    """
    missing = [k for k in key_columns if k not in schema.names]
    if missing:
        raise BackendError(
            f"key_columns {missing} are not columns of the data being written "
            f"({schema.names}), so the table cannot be created with them as its key."
        )
    parts = [
        f"{quote(f.name, dialect)} {sql_type(f.type, dialect)}"
        + (" NOT NULL" if f.name in key_columns else "")
        for f in schema
    ]
    if key_columns:
        keys = ", ".join(quote(c, dialect) for c in key_columns)
        parts.append(f"PRIMARY KEY ({keys})")
    return f"CREATE TABLE {qualified_table(table, dialect)} ({', '.join(parts)})"
