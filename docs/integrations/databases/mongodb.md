# MongoDB

This page covers reading a MongoDB collection into Arrow and writing a dataset back as bulk upserts. Both directions need `pip install 'batcher-engine[mongo]'`, which brings `pymongo` and `pymongoarrow`.

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.mongo(uri=..., database=..., collection=...) <batcher.api.io_namespace.reader.Reader.mongo>` |
| **Write** | {py:meth}`ds.write.mongo(collection, uri=..., database=..., mode=...) <batcher.api.io_namespace.writer.Writer.mongo>`, one bulk round trip: upsert, append, overwrite, or delete |
| **Extra** | `pip install 'batcher-engine[mongo]'` |
| **Parallelism** | Off by default. `PartitionSpec(segments=N)` splits the `_id` range. |
| **Pushdown** | Predicates become a Mongo filter document, AND-merged into the `find` |
| **Credentials** | In the URI, which is never logged. The whole URI may be an `env:`/`file:` reference |

Reads go through `pymongoarrow.api.find_arrow_all`, which builds an Arrow table directly from the
wire, with no `dict` per document and no per-row Python. That is what makes a Mongo scan practical at analytical scale, and it is also why this connector needs your documents to have a stable shape.

## Reading

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

The URI carries the credentials and is never logged. Pass `uri="env:MONGO_URI"` to keep it out of the driver process entirely. The reference is resolved where the connection is dialed.

Learned statistics live under the connector's `identity()`, which is `mongo:<database>.<collection>:<fingerprint>`. The fingerprint is a `sha256` over the connection options with the URI's password masked. The same collection on staging and on production therefore keeps separate statistics, and rotating the password neither leaks into the key nor orphans what has already been learned.

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

## Writing

`ds.write.mongo(collection, uri=..., database=...)` applies every row in one `bulk_write` per batch, never a per-row round trip. `mode` says what it applies:

| Mode | Effect |
| --- | --- |
| `upsert` (default) | Replace the document holding the same `key_field`, or insert if absent. |
| `append` | Insert every row. An `_id` collision is an error rather than a silent replace. |
| `overwrite` | Empty the collection, then insert. |
| `delete` | Remove the documents named by `key_field`. |

`mode` defaults to `upsert` rather than to `ds.write`'s usual `overwrite`: a collection is a store that is maintained rather than replaced, and defaulting to the destructive mode would empty it on a call that never said so.

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

"Upsert on a key" buys idempotency: re-running the job replaces the same documents rather than
duplicating them, which is what makes a retried or recomputed partition safe. It also costs
something, because it is a *replace* rather than a field merge.

:::{important}
The matched document is replaced wholesale by the row, so columns you did not select are not
preserved.
:::

Rows do cross into Python for the write ({py:meth}`to_pylist() <batcher.Dataset.to_pylist>` per batch, then one `bulk_write`). That is
the driver's shape, and it makes Mongo a fine sink for a serving or feature collection and a poor
one for dumping a billion analytical rows. Write those to Parquet or Delta.

## Requirements and limitations

Schema inference reads one document. The Arrow schema comes from a `limit=1` sample, so a collection whose documents disagree gives a schema that does not describe it. A field that is an `int` in some documents and a `string` in others, or one missing from the sampled document, makes later batches fail to convert or arrive null. Constrain the shape with an explicit `query=`, or project the fields you need.

Range splitting assumes `_id` is ordered and comparable, as an ObjectId is. A collection with a custom, unordered `_id` such as a random UUID string still splits, but the ranges will not be balanced.

A slow downstream pipeline holds each split's cursor open while the engine consumes it. If a long job hits cursor-not-found errors, lower `segments` so each cursor drains faster rather than raising the server timeout.

The bulk write is `ordered=False` with no commit phase. A write that fails halfway leaves the documents it already applied in place, so recovery means re-running it, which a keyed `upsert` makes safe.

`overwrite` is single-node only. Past the first shard of a distributed write it is refused, because every shard would empty the one collection they all target and discard the shards before it. Distribute an `upsert` instead, which only touches the keys its own rows name.

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>`: sources, splits, pushdown.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: sinks, modes, and idempotent re-runs.
- {doc}`Writing to a database </integrations/databases/writing>`: the same mode vocabulary across SQL and the operational stores.
- {doc}`Feature pipeline </cookbook/ml/pipelines/features/feature-pipeline>`: the shape that ends in an upsert to
  a serving collection.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the {py:class}`Source <batcher.io.Source>`/{py:class}`Sink <batcher.io.Sink>`/{py:class}`Split <batcher.io.Split>` protocol.
- {doc}`I/O API </api/relational/io>`: the full reader/writer reference.
- {doc}`Elasticsearch </integrations/databases/elasticsearch>`: the other document store, with the same write vocabulary.
