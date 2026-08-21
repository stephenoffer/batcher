# Key-value stores

This page covers DynamoDB, Cassandra (and ScyllaDB), Redis, and HBase: the stores Batcher reads and writes by *key* rather than by query. It explains how a full read parallelizes, when a filtered read stops being a full read at all, and what each store's write path can and cannot express.

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.dynamodb(table=...) <batcher.api.io_namespace.reader.Reader.dynamodb>`, {py:meth}`bt.read.cassandra(...) <batcher.api.io_namespace.reader.Reader.cassandra>`, {py:meth}`bt.read.redis(...) <batcher.api.io_namespace.reader.Reader.redis>`, {py:meth}`bt.read.hbase(...) <batcher.api.io_namespace.reader.Reader.hbase>` |
| **Write** | {py:meth}`ds.write.dynamodb(table, ...) <batcher.api.io_namespace.writer.Writer.dynamodb>`, {py:meth}`ds.write.cassandra(table, ...) <batcher.api.io_namespace.writer.Writer.cassandra>`, {py:meth}`ds.write.redis(prefix, ...) <batcher.api.io_namespace.writer.Writer.redis>`, {py:meth}`ds.write.hbase(table, ...) <batcher.api.io_namespace.writer.Writer.hbase>` |
| **Extras** | `pip install 'batcher-engine[dynamodb]'`, `[cassandra]`, `[redis]`, `[hbase]` |
| **Parallelism** | Scan segments, token ranges, hash-slot ranges, region ranges — one split each |
| **Credentials** | Passed as connection keywords, never logged, and `env:`/`file:` references resolve on the worker |

## Reading a whole store

Each store has a native parallel unit, and Batcher maps one split onto each of them rather than inventing client-side range math.

| Store | Parallel unit | Default |
| --- | --- | --- |
| DynamoDB | A `Scan` segment (`Segment` / `TotalSegments`) | one segment |
| Cassandra | A Murmur3 token range | 64 ranges |
| Redis | A contiguous hash-slot range of the 16,384 | one range |
| HBase | A region's ``[start_key, stop_key)`` range | one per region |

Raise the count with `partition_spec=PartitionSpec(segments=N)`. On Cassandra the default of 64 is already about one range per vnode; on DynamoDB the default is one, because segment count is read capacity you are choosing to spend.

## A filter can stop the fan-out entirely

A parallel scan is the right shape for reading a table. It is the wrong shape for reading one row, and a server-side filter does not fix that.

On DynamoDB it makes it worse than it looks. A `FilterExpression` is applied *after* items are read, and read capacity is billed for what was examined rather than for what came back. So a scan filtered down to one item costs the same as reading the table.

Both stores have an operation that avoids it, and Batcher reaches for it when the pushed predicate proves it can:

| Store | Ordinary read | When the partition key is pinned |
| --- | --- | --- |
| DynamoDB | N parallel `Scan` segments with a `FilterExpression` | one `Query` with a `KeyConditionExpression` |
| Cassandra | 64 token-range `SELECT`s with `ALLOW FILTERING` | one `SELECT` with no token predicate |

```python
# docs: skip
import batcher as bt
from batcher import col

events = bt.read.dynamodb(table="events", region_name="us-east-1")

# One Query against one partition. Not a scan.
recent = events.filter(col("user_id") == "u-42", col("ts") > 1_700_000_000)
```

This is about **cost**, not latency. Reading one DynamoDB partition instead of scanning the
table is the difference between one read unit and the whole table's worth of them, and on
Cassandra between one replica set and the entire ring. It does not make Batcher a serving
path: the query still carries the engine's fixed ~2 ms floor, and one process serves a few
hundred lookups a second. See {doc}`SQL databases </integrations/databases/databases>` for
the measured numbers.

### What "pinned" means, and why the rule is strict

The rewrite is sound only when reading the one partition cannot miss a matching row. That holds exactly when the predicate has a top-level `AND` term of the form `partition_key = <literal>`: every row satisfying the predicate then carries that key value, so nothing that matches lives anywhere else.

Three shapes look close and are not, and each falls back to the full scan:

- **A top-level `OR`.** One branch pinning the key says nothing about the other, which can match a row in any partition.
- **A range on the partition key.** `user_id > "a"` names no partition; the key is hashed, so ordering on it means nothing to the store.
- **A composite partition key only partly pinned.** Cassandra hashes the whole key together, so fixing one of two columns still names no partition.

Everything the predicate says beyond the key becomes a `FilterExpression` on DynamoDB, or stays in the `WHERE` on Cassandra. A term that will not translate is simply left to the engine, which re-checks every row regardless.

### Tell DynamoDB its keys, or let it ask

Batcher learns the key schema from `DescribeTable`. A role granted `dynamodb:Query` and `dynamodb:Scan` but not `dynamodb:DescribeTable` is a common least-privilege split, and there the metadata call fails and the read stays a scan. Pass the keys instead:

```python
# docs: skip
events = bt.read.dynamodb(
    table="events",
    region_name="us-east-1",
    partition_key="user_id",
    sort_key="ts",
)
```

Cassandra always needs `partition_key=` anyway, because the token predicate is built from it.

### Temporal predicates are not pushed

Neither DynamoDB nor Elasticsearch has a date type of its own: an application stores a timestamp as an ISO string, as epoch seconds, or as epoch millis, and nothing in the data says which. A comparison against a date or a timestamp is therefore evaluated by the engine rather than pushed, which costs bandwidth and never rows. Store a key you can compare — an epoch integer, or an ISO string with a fixed width — if you need the server to narrow on time.

## Writing

The write vocabulary is the same one the {doc}`SQL sink </integrations/databases/writing>` uses, and each store implements the part of it that it can express.

| Store | Modes | Bulk primitive |
| --- | --- | --- |
| DynamoDB | `upsert`, `delete` | `BatchWriteItem`, 25 requests per call |
| Cassandra | `upsert`, `delete` | one prepared statement, run concurrently |
| Redis | `upsert`, `delete` | one pipeline per batch |
| HBase | `upsert`, `delete` | one happybase `Batch` per Arrow batch |

```python
# docs: skip
scores.write.dynamodb("user_scores", region_name="us-east-1")
features.write.cassandra("features", contact_points=["c1"], keyspace="serving")
sessions.write.redis("session", host="cache", ttl_seconds=3600)
```

`append` is missing from all four because none of them can express it. A DynamoDB `PutItem` replaces the item holding the same key, and no batch operation inserts only when the key is absent, so an "append" would silently be an upsert. A CQL `INSERT` and an HBase `Put` are upserts for the same reason. Redis `SET` replaces.

`overwrite` is missing because emptying these stores is not a write. On DynamoDB it means scanning the table to delete every item at full read and write cost; on Cassandra it means `TRUNCATE`, a cluster-wide schema operation; on Redis it means `FLUSHDB`, which discards every key in the database rather than the ones this write would replace; on HBase it means disabling and truncating the table through the admin API. Reaching an operation of that reach by passing a string to `mode` is not something a write API should offer. Write to a new table or key prefix and re-point what reads it.

### What HBase writes

Every column but the row key becomes a cell. A column named `family:qualifier` keeps its family, and one without a colon is placed in `column_family=` (default `"cf"`). That is what lets a frame read by `bt.read.hbase(...)` — whose column names already carry the family — round-trip unchanged, while a plain relational frame can be written without qualifying every column by hand. A null cell is left unwritten rather than stored as the four characters `None`.

### What Redis writes

The shape of the frame decides, and the rule is the one that makes a round trip return what was written:

- Two columns named `key` and `value` — the shape `bt.read.redis(...)` returns — writes one string per key.
- Anything wider writes one hash per key, with a field per remaining column.

Keys are prefixed by `prefix=` if given, and by the write's destination name otherwise, so `ds.write.redis("session")` writes `session:<key>`. Nulls are written as the empty string rather than the four characters `None`.

### Partial success is checked, not assumed

Each of these APIs can fail *inside* a successful call, and each sink reads the response rather than the status.

`BatchWriteItem` returns the requests it could not process under `UnprocessedItems` with a 200 — throttling, usually. Those are resent with jittered backoff, and a remainder that survives every attempt raises rather than being dropped. Cassandra's concurrent execution returns a success flag per statement; a failed one raises naming how many failed and what the first said.

A sink that trusted the call would have written some of its rows and reported success, which is the quietest kind of data loss there is.

## Requirements and limitations

A distributed write is one operation per shard, with no transaction across them. `upsert` and `delete` are safe that way, because a shard only ever touches the keys its own rows name.

An HBase `Put` replaces only the cells it names and leaves the rest of the row alone, so an upsert there is a *merge* of columns rather than a replacement of the row. That differs from DynamoDB and Cassandra, where the write replaces the item or row, and it matters when two pipelines maintain different columns of the same key.

Schema inference samples one item or one row. A store whose records disagree — a field that is a number in some and a string in others, or missing from the first — gives a schema that does not describe the store, and later batches then fail to convert or arrive null. Narrow the read with a `query=` that constrains the shape, or project the fields you need.

Rows cross into Python on both the read and the write for all three stores. That is the drivers' shape, and it makes these good sinks for a serving or feature dataset and poor ones for moving a billion analytical rows. Write those to Parquet or a lakehouse table.

Redis reads every matching key's value with a round trip per key inside its slot range. `match=` is what keeps that bounded; a read with no pattern walks the whole keyspace.

## See also

- {doc}`Writing to a database </integrations/databases/writing>`: the same mode vocabulary, and the SQL sinks it came from.
- {doc}`MongoDB </integrations/databases/mongodb>` and {doc}`Elasticsearch </integrations/databases/elasticsearch>`: the two document stores, with their own read paths.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: sources, splits, and how a pushed predicate reaches a connector.
- {doc}`I/O API </api/relational/io>`: the full reader and writer reference.
