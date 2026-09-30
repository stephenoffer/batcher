# How a table is named

This page covers the name a governance policy is keyed on: what each source is named by,
and which spellings of the same object fold together.

It continues {doc}`Governance and security </user-guide/trust/governance>`. Getting the name wrong
is not cosmetic. A policy matched against a name the reader never produces
governs nothing at all, silently, and no error says so.

```python
import batcher as bt

analyst = bt.Principal("ana", roles=["analyst"])
```

## What each source is named by

A policy is keyed on the table, and the table is named by what an operator can write down
before anyone has read it:

| Source | Named by |
|---|---|
| Files and directories (Parquet, CSV, JSON, text, ...) | the path |
| Delta, Delta change feed, Hudi | the table URI |
| Iceberg | the table identifier |
| Kafka, Kinesis, Pulsar, Event Hubs, Pub/Sub | the topic, stream, or subscription |
| A SQL query (Snowflake, BigQuery `query=`, `bt.read.sql`) | nothing, unless the read declares `governed_as`; see below |
| In-memory tables, a rate generator, a raw socket | nothing; see below |

The name is the **table**, never the slice of it a particular query reads. A read narrowed
by `n_rows`, `columns`, or an explicit file list is the same table, and so is a Delta table
read at an older version. That distinction is load-bearing: an engine that keyed policy on
the slice would leave `bt.read.parquet(path, n_rows=2)` governed by nothing.

An in-memory table and a live socket have no durable name, so no policy can be declared
about them. `governance.mode` decides what to do about that. Under `strict` such a read is
refused rather than exempted.

## Declare the table a query reads

A read defined by SQL names no table. Batcher doesn't parse the query to guess one, because a policy matched against the wrong table governs the wrong data. Declare the name instead with `governed_as`, spelled exactly as the policy spells it, and that table's policy is applied to the query's result:

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE users (id INTEGER, ssn TEXT)")
con.execute("INSERT INTO users VALUES (1, '111')")
catalog = bt.SecurityCatalog().grant("analyst", on="main.users", select=["id"])

with bt.security(catalog, analyst):
    users = bt.read.sql("SELECT * FROM users", connection=con, governed_as="main.users")
    print(users.to_pydict())
```

The declaration is matched against the result's column names, so a query that renames a governed column (`SELECT ssn AS x`) escapes a mask keyed on `ssn`. A grant that lists the visible columns still withholds `x`, because a column no grant names isn't visible.

Inside a {py:obj}`bt.security() <batcher.security>` block, an undeclared query whose text names a governed table, either in full or by its last dotted component, is refused with `AccessDeniedError` rather than read ungoverned. That check is a safety net rather than the guarantee. It reads identifiers written in the query, so a view or a synonym over a governed table passes it. A declaration that differs from a governed name only in case is refused too, since a warehouse identifier is usually case-insensitive and the policy name isn't. A source that names its own table, such as a Parquet path, can't be re-declared under a different name.

## One object, one policy

The same file has many spellings. `s3a://` is the Hadoop spelling of `s3://`, a trailing
slash names the same directory as no trailing slash, and both `bucket//key` and
`bucket/tmp/../key` name `bucket/key`. A policy matched on the raw string would fire on one
spelling and not the rest. Every alias would be a bypass.

Batcher folds them to one canonical name on the way in and on the way out. Declare a
policy in whichever spelling your catalog uses, and it governs every read of that object.

```python
aliased = bt.SecurityCatalog().grant("analyst", on="s3://vault/pii.parquet", select=["id"])

print(aliased.visible_columns("s3a://vault/pii.parquet", ["id", "ssn"], analyst))
```

The bucket and account are left exactly as written. Case is significant in an S3 key and
in an ABFS container, so folding it would merge two objects that are genuinely different.

## See also

- {doc}`Governance and security </user-guide/trust/governance>`: the catalog these names key.
- {doc}`Write privileges </user-guide/trust/write-privileges>`: the same names, matched for a write.
- {doc}`Hardening a deployment </user-guide/trust/hardening>`: `governance.mode="strict"`, which refuses a source with no durable name.
