# How a table is named

This page covers the name a governance policy is keyed on: what each source is named by,
and which spellings of the same object fold together.

It continues {doc}`Governance and security </user-guide/trust/governance>`. Getting this
wrong is not a cosmetic problem: a policy matched against a name the reader never
produces silently governs nothing, and no error says so.

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
| In-memory tables, a rate generator, a raw socket | nothing -- see below |

The name is the **table**, never the slice of it a particular query reads. A read narrowed
by `n_rows`, `columns`, or an explicit file list is the same table, and so is a Delta table
read at an older version. That distinction is load-bearing: an engine that keyed policy on
the slice would leave `bt.read.parquet(path, n_rows=2)` governed by nothing.

An in-memory table and a live socket have no durable name, so no policy can be declared
about them. `governance.mode` decides what to do about that -- `strict` refuses such a read
rather than exempting it.


## One object, one policy

The same file has many spellings. `s3a://` is the Hadoop spelling of `s3://`, a trailing
slash names the same directory as no trailing slash, and `bucket//key` and
`bucket/tmp/../key` name `bucket/key`. A policy that matched the string would fire on one
and not the others, so every alias was a bypass.

Batcher folds them to one canonical name on the way in and on the way out. Declare a
policy in whichever spelling your catalog uses, and it governs every read of that object.

```python
aliased = bt.SecurityCatalog().grant("analyst", on="s3://vault/pii.parquet", select=["id"])

print(aliased.visible_columns("s3a://vault/pii.parquet", ["id", "ssn"], analyst))
```

The bucket and account are left exactly as written. Case is significant in an S3 key and
in an ABFS container, so folding it would merge two objects that are genuinely different.

