# Vendor capability matrix

This page states what Batcher knows about the databases it is most often pointed at: which route serves each one, the package that installs the route, the write modes the route carries, and what happens to the vendor types that do not map cleanly onto Arrow. A generic route reaching a database is not the same as that database being covered, and this page is where the difference is written down.

:::{warning}
Nothing on this page has run against a live server yet, which is what the `untested` status means. The rules marked "Batcher, unit-tested" are enforced by Batcher's own code and pinned by unit tests against fake drivers. The rules marked "Stated, not enforced" describe documented driver behaviour that Batcher passes through. See [`tests/PENDING_VERIFICATION.md`](https://github.com/stephenoffer/batcher/blob/main/tests/PENDING_VERIFICATION.md).
:::

## Routes and write modes

Each vendor is reached by an Arrow-native reader first and a PEP 249 driver second, as {doc}`databases` describes. The DB-API route is the only one that executes statements, so it is the one that carries the row-level modes. The upsert column is read from the statement builder the sink uses, so it reports the SQL Batcher would send. The following table is ordered by vendor, then by route preference:

| Vendor | Route | Package | Read | Write modes | Upsert | Status |
| --- | --- | --- | --- | --- | --- | --- |
| PostgreSQL | adbc | `adbc-driver-postgresql` | yes | append, overwrite | n/a | untested |
| PostgreSQL | dbapi | `psycopg[binary]` | yes | append, overwrite, upsert, update, delete, delete_insert | on_conflict | untested |
| MySQL / MariaDB | connectorx | `connectorx` | yes | read-only | n/a | untested |
| MySQL / MariaDB | dbapi | `pymysql` | yes | append, overwrite, upsert, update, delete, delete_insert | on_duplicate_key | untested |
| SQL Server | connectorx | `connectorx` | yes | read-only | n/a | untested |
| SQL Server | dbapi | `pymssql` | yes | append, overwrite, upsert, update, delete, delete_insert | merge | untested |
| Oracle | connectorx | `connectorx` | yes | read-only | n/a | untested |
| Oracle | dbapi | `oracledb` | yes | append, overwrite, upsert, update, delete, delete_insert | merge | untested |
| Trino | connectorx | `connectorx` | yes | read-only | n/a | untested |
| Trino | dbapi | `trino` | yes | append, overwrite | n/a | untested |
| Redshift | connectorx | `connectorx` | yes | read-only | n/a | untested |
| Redshift | dbapi | `redshift-connector` | yes | append, overwrite, upsert, update, delete, delete_insert | merge | untested |

When no driver for a route is installed, the error names every route's package and what it buys, so `pip install pymysql` is suggested for a MySQL upsert rather than a reader that cannot write.

## Type rules

Three problems recur across vendors. An unsigned 64-bit integer can exceed the engine's `int64` range. A driver can hand back a value of the wrong Python type, such as PyMySQL's zero date returned as a string. And a driver can return a lossy type by default, such as python-oracledb's float for a fractional `NUMBER`. Batcher either converts each one in a declared way or refuses it with an error that names the column and the fix.

The opt-in conversions are keywords on the read, accepted by `bt.read.sql` and `bt.read.table("dbapi", ...)`:

```python
# docs: skip
import batcher as bt

orders = bt.read.sql(
    "SELECT id, shipped_on FROM orders",
    uri="mysql://svc@db/shop",
    unsigned="decimal",
    zero_dates="null",
)
```

`unsigned="decimal"` reads every `uint64` column as `decimal128(20, 0)`, on the ConnectorX and DB-API routes. `zero_dates="null"` reads a MySQL zero date as NULL on the DB-API route. `oracle_numbers="decimal"` installs python-oracledb's documented output type handler, so every `NUMBER` arrives as an exact decimal.

### PostgreSQL

The following table lists the PostgreSQL type and mode rules. The live smoke test reads `BATCHER_LIVE_POSTGRES_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| NUMERIC 'NaN' / 'Infinity' (DB-API) | Batcher, unit-tested | Refused naming the column: an Arrow decimal cannot hold them. CAST the column to double precision in the query to keep them. |
| NUMERIC (DB-API) | Stated, not enforced | Python Decimal values become decimal128, or decimal256 past 38 digits, sized to the values in the first batch. |
| ARRAY (DB-API) | Stated, not enforced | Python lists become list<T>; a multi-dimensional array nests as list<list<T>>. |
| TIMESTAMPTZ (DB-API) | Stated, not enforced | Timezone-aware values become timestamp[us, tz] with every instant preserved. |

### MySQL / MariaDB

The following table lists the MySQL / MariaDB type and mode rules. The live smoke test reads `BATCHER_LIVE_MYSQL_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| BIGINT UNSIGNED | Batcher, unit-tested | Arrives as uint64. A value above 2^63-1 is refused naming the column; pass unsigned='decimal' to read the column as decimal128(20, 0) instead. |
| DATE / DATETIME '0000-00-00' (DB-API) | Batcher, unit-tested | PyMySQL returns a zero date as a string. Refused naming the column by default; pass zero_dates='null' to read it as NULL. mysqlclient already returns None for it. |
| JSON | Stated, not enforced | Arrives as a string column; extract fields with the .json accessor. |

### SQL Server

The following table lists the SQL Server type and mode rules. The live smoke test reads `BATCHER_LIVE_MSSQL_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| DECIMAL / NUMERIC (DB-API) | Stated, not enforced | pymssql returns Python Decimal, which becomes an exact Arrow decimal. |
| NVARCHAR / NCHAR (DB-API) | Stated, not enforced | Decoded to string by pymssql, whose connection charset defaults to UTF-8. |
| DATETIME2(7) | Stated, not enforced | Python datetime holds microseconds, so the seventh fractional digit is truncated by the driver. |
| Bulk writes | Stated, not enforced | Rows are bound through executemany; there is no BCP path. staged=True makes a distributed append or overwrite atomic. |

### Oracle

The following table lists the Oracle type and mode rules. The live smoke test reads `BATCHER_LIVE_ORACLE_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| NUMBER with a fractional scale (DB-API) | Batcher, unit-tested | python-oracledb returns a float by default. Pass oracle_numbers='decimal' to fetch every NUMBER as an exact Decimal. |
| TIMESTAMP WITH TIME ZONE (DB-API) | Stated, not enforced | python-oracledb returns a naive datetime, dropping the offset; select SYS_EXTRACT_UTC(column) to read the instant. |
| CLOB / BLOB (DB-API) | Batcher, unit-tested | Returned as LOB handles unless the driver is configured to fetch them as str/bytes; an unconvertible value is refused naming the column and type. |

### Trino

The following table lists the Trino type and mode rules. The live smoke test reads `BATCHER_LIVE_TRINO_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| Authentication | Batcher, unit-tested | A password in the URI or password= becomes the client's BasicAuthentication on the worker, over https unless http_scheme is set. |
| Catalog and schema | Batcher, unit-tested | trino://user@host:443/catalog/schema sets the session catalog and schema. |
| Row-level writes | Stated, not enforced | UPDATE, DELETE and MERGE depend on the catalog connector, so only append and overwrite are listed; upsert has no Trino spelling in Batcher. |

### Redshift

The following table lists the Redshift type and mode rules. The live smoke test reads `BATCHER_LIVE_REDSHIFT_URI`.

| Vendor type or mode | Applied by | Rule |
| --- | --- | --- |
| Upsert | Stated, not enforced | Spelled as MERGE, a statement shape not yet run against Redshift. mode='delete_insert' uses only DELETE and INSERT. |

## See also

- {doc}`databases`: connection URIs, routing and parallel extraction.
- {doc}`writing`: the write modes and how each shard commits.
- {doc}`/integrations/warehouses/index`: Snowflake, BigQuery, Databricks and Athena.
