# MongoDB

Read a collection into Arrow, write a dataset back as bulk upserts. Both directions need
`pip install 'batcher-engine[mongo]'`, which brings `pymongo` and `pymongoarrow`.

| | |
| --- | --- |
| **Read** | `bt.read.mongo(uri=..., database=..., collection=...)` |
| **Write** | `ds.write.mongo(collection, uri=..., database=...)`, a bulk upsert on `key_field` |
| **Extra** | `pip install 'batcher-engine[mongo]'` |
| **Parallelism** | Off by default. `PartitionSpec(segments=N)` splits the `_id` range. |
| **Pushdown** | Predicates become a Mongo filter document, AND-merged into the `find` |
| **Credentials** | In the URI, which is never logged |

Reads go through `pymongoarrow.api.find_arrow_all`, which builds an Arrow table directly from the
wire. No per-row Python, no `dict` per document. That is the only reason a Mongo scan is worth
doing at analytical scale, and it is why this connector cares about your documents having a
*stable* shape.

## Read

::::{tab-set}

:::{tab-item} The whole collection

```python
# docs: skip
import batcher as bt

events = bt.read.mongo(
    uri="mongodb://user:pass@mongo.internal:27017",
    database="app",
    collection="events",
)
print(events.filter(bt.col("status") == "active").count())
```
:::

:::{tab-item} With a server-side filter

`query=` takes a Mongo filter document applied to every read, on top of whatever the optimizer
pushes down:

```python
# docs: skip
recent = bt.read.mongo(
    uri="mongodb://mongo.internal:27017",
    database="app",
    collection="events",
    query={"created_at": {"$gte": "2024-01-01"}},
)
```
:::

::::

The URI carries the credentials. It is stored verbatim on the source and never logged: the
connector's `identity()` is `mongo:<database>.<collection>`, deliberately free of the connection
string, so a plan dump or a log line cannot leak your password.

## Predicate pushdown

The pushable part of the query's `WHERE` becomes a Mongo filter document and is AND-merged into the
`find`, so the server prunes before anything is serialized.

:::{dropdown} See the translation, with no server running
```python
import batcher as bt
from batcher.io.predicate import to_mongo_filter

predicate = ((bt.col("status") == "active") & (bt.col("amount") > 100)).to_ir()
print(to_mongo_filter(predicate))
# {'$and': [{'status': {'$eq': 'active'}}, {'amount': {'$gt': 100}}]}
```
:::

What cannot be expressed as a filter document is not pushed, and the engine's own filter
produces the same rows from a wider scan. Correctness never depends on the push. Throughput does.
Index the fields you filter on, or the server does a collection scan and the pushdown buys you
nothing but a smaller result.

## How it parallelizes

A parallel read splits the `_id` key space into contiguous half-open `[lo, hi)` ObjectId ranges,
one split per range, each issuing its own bounded `find`. The boundaries are sampled by sorted
offset so the ranges hold comparable row counts, and they are a disjoint, exhaustive cover: no
document is read twice, none is missed.

Parallelism is off by default (one split). Ask for it with a `PartitionSpec`:

```python
# docs: skip
import batcher as bt
from batcher.io.formats.nosql import PartitionSpec

events = bt.read.mongo(
    uri="mongodb://mongo.internal:27017",
    database="app",
    collection="events",
    partition_spec=PartitionSpec(segments=16),
)
```

:::{warning}
Sixteen segments means sixteen concurrent cursors against your cluster. That is a load decision as
much as a throughput one. On a replica set serving production traffic, read from a secondary, and
do not casually set `segments` to your core count on a shared cluster.
:::

Boundary sampling itself costs a `count_documents` plus one `find(...).skip(offset).limit(1)` per
boundary. On a huge collection those `skip`s are not free, which is why `segments` should be tens,
not thousands.

`count()` is answered by `count_documents` with the pushed filter, so an unfiltered count moves no
documents.

## Write

`ds.write.mongo(collection, uri=..., database=...)` upserts every row, keyed on `key_field`
(default `_id`), in one `bulk_write` per batch. Not a per-row round trip.

```python
# docs: skip
import batcher as bt

scored = bt.read.parquet("s3://lake/scores/*.parquet")
scored.write.mongo(
    "scores",
    uri="mongodb://mongo.internal:27017",
    database="app",
    key_field="user_id",
)
```

Two things follow from "upsert on a key". First, the write is idempotent: re-running the job
replaces the same documents rather than duplicating them, which is what makes a retried or
recomputed partition safe. Second, it is a *replace*, not a field merge.

:::{important}
The matched document is replaced wholesale by the row, so columns you did not select are not
preserved.
:::

Rows do cross into Python for the write (`to_pylist()` per batch, then one `bulk_write`). That is
the driver's shape, and it makes Mongo a fine sink for a serving or feature collection and a poor
one for dumping a billion analytical rows. Write those to Parquet or Delta.

## Failure modes worth knowing

:::{warning}
**Schema inference reads one document.** The Arrow schema comes from a `limit=1` sample. A
collection whose documents disagree, where a field is an `int` in some rows and a `string` in
others, or missing entirely from the first document, gives you a schema that does not describe the
collection, and later batches then fail to convert or arrive null. Fix it with an explicit `query=`
that constrains the shape, or project the fields you actually need. Mongo's freedom of shape is
exactly the thing an Arrow reader cannot absorb.
:::

**`_id` is an ObjectId.** It arrives as its Arrow-mapped type, and range splitting assumes `_id` is
ordered and comparable. A collection with a custom, unordered `_id` (a random UUID string) still
splits, but the ranges will not be balanced.

**Cursor timeouts.** A slow downstream pipeline holds each split's cursor open while the engine
consumes it. If you see cursor-not-found errors on a long job, the fix is a smaller `segments`
count with a faster drain, not a bigger timeout.

**No transactions.** The bulk upsert is `ordered=False` and there is no commit phase. A write that
fails halfway leaves the documents it already upserted in place. Idempotency on the key is your
recovery story; there is no rollback.

## See also

- [Reading data](../user-guide/reading-data.md): sources, splits, pushdown.
- [Writing data](../user-guide/writing-data.md): sinks, modes, and idempotent re-runs.
- [Feature pipeline](../examples/ml/feature-pipeline.md): the shape that ends in an upsert to
  a serving collection.
- [Custom connectors](../user-guide/custom-connectors.md): the `Source`/`Sink`/`Split` protocol.
- [I/O API](../api/io.md): the full reader/writer reference.
- [Elasticsearch](elasticsearch.md): the other document store, read-only.
