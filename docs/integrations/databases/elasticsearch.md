# Elasticsearch

This page covers reading an Elasticsearch index into the engine and indexing rows back into one. The read takes an ES|QL result as a single Arrow stream, or slices a scroll across workers when you need raw documents. The write sends `_bulk` requests and checks every item in the response, which is the part an application-side loop usually gets wrong.

Batcher indexes into an index you manage. It never creates or changes a mapping, and it never forces a refresh unless you ask.

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.elasticsearch(hosts=..., index=..., esql=...) <batcher.api.io_namespace.reader.Reader.elasticsearch>` |
| **Write** | {py:meth}`ds.write.elasticsearch(index, hosts=...) <batcher.api.io_namespace.writer.Writer.elasticsearch>`, over `_bulk` |
| **Extra** | `pip install 'batcher-engine[elasticsearch]'` (ES 8.18+ for the Arrow path) |
| **Parallelism** | Sliced scroll: one split per slice. The ES\|QL path is a single split. |
| **Pushdown** | Predicates become an appended `\| WHERE` (ES\|QL) or a `bool` query (scroll) |
| **Credentials** | `hosts` and `api_key`, stored on the source and never logged |

## Two read paths, and they are not equivalent

::::{tab-set}

:::{tab-item} ES|QL with Arrow output

This is the one you want. Elasticsearch 8.18+ can return an ES|QL result as an Arrow stream, which
Batcher reads straight into `RecordBatch`es with no per-row Python. The cluster does the filtering
and the aggregation, and columns come back.

```python
# docs: skip
import batcher as bt

logs = bt.read.elasticsearch(
    hosts="https://es.internal:9200",
    index="logs-*",
    api_key="...",
    esql="FROM logs-* | WHERE status >= 500 | KEEP @timestamp, service, status, latency_ms",
)
print(logs.group_by("service").agg(bt.col("latency_ms").mean().alias("p_mean")).to_pydict())
```
:::

:::{tab-item} Sliced scroll

The fallback when you don't pass `esql=`. It runs a DSL query, scrolls the hits, and assembles
Arrow from each hit's `_source`. That means JSON documents through Python before they become
columns. It works, it is much slower, and it is where you land by accident if you omit `esql=`.

```python
# docs: skip
import batcher as bt

hits = bt.read.elasticsearch(
    hosts="https://es.internal:9200",
    index="logs-2024.03.01",
    query={"range": {"status": {"gte": 500}}},
)
```
:::

::::

:::{warning}
Omitting `esql=` silently drops you onto the scroll path, with a JSON document per hit crossing
Python before it becomes a column.
:::

## Credentials

`hosts` and `api_key` are stored on the source and never logged. `api_key` also accepts an `env:`/`file:` reference, resolved on the worker that opens the connection.

The connector's identity, which keys its learned statistics, is `elasticsearch:<index>:<fingerprint>`. The fingerprint is a `sha256` over the connection options with credential keys excluded, so `orders` on staging and `orders` on production keep separate statistics.

Use an API key scoped to the indices you read. A scroll holds a cursor open on the cluster, and that key should not be able to do anything else.

## Predicate pushdown

Kyber pushes the query's filter into the cluster, and how depends on the path. On the ES|QL path it
becomes an appended `| WHERE` clause. On the scroll path it becomes an ES `bool` query AND-merged
with your DSL query.

:::{dropdown} See the ES|QL translation, with no cluster running
```python
import batcher as bt
from batcher.io.predicate import to_sql_where

predicate = ((bt.col("status") == "active") & (bt.col("bytes") > 1000)).to_ir()
print(to_sql_where(predicate))
# (status = 'active' AND bytes > 1000)
```
:::

Column-vs-literal comparisons, `IS NULL` / `IS NOT NULL`, and `AND`/`OR` of those push. Anything
else (a column-vs-column comparison, a computed term) does not push, the read stays wide, and the
engine's own filter still produces the correct rows.

## How it parallelizes

The scroll uses Elasticsearch's own sliced scroll: a search declares `slice = {id, max}` and each
slice covers a disjoint subset of the matching documents. Batcher makes one split per slice, so
`segments` is your read fan-out.

```python
# docs: skip
import batcher as bt
from batcher.io.formats.nosql import PartitionSpec

hits = bt.read.elasticsearch(
    hosts="https://es.internal:9200",
    index="logs-*",
    partition_spec=PartitionSpec(segments=8),
)
```

The ES|QL path does not slice. The whole result comes back in one Arrow stream, as a single
split. That is the tradeoff, and it is usually the right one: ES|QL pushes the filter and the
projection into the cluster, so what crosses the wire is small, and one fast stream of a small
result beats eight slow scrolls of a large one. Reach for slices when you genuinely need to pull a
large, unaggregated slab of documents out.

:::{tip}
Sizing slices above the index's shard count buys nothing, because a slice cannot span a shard.
Match `segments` to shards, not to your CPU count.
:::

## Writing

`ds.write.elasticsearch(index, hosts=...)` sends one `_bulk` request per 1,000 documents.

| Mode | Effect |
| --- | --- |
| `upsert` (default) | Index each document under `key_field`'s value as its `_id`, replacing what was there. |
| `append` | Index without an `_id`, so the cluster assigns one. |
| `overwrite` | `delete_by_query` the whole index, then index. |
| `delete` | Delete the documents named by `key_field`. |

```python
# docs: skip
scored.write.elasticsearch(
    "products",
    hosts="https://es.internal:9200",
    api_key="env:ES_API_KEY",
    key_field="sku",
)
```

Every response is read item by item. `_bulk` reports per-document failures inside an HTTP 200: a mapping conflict on one document leaves the other 999 indexed and the call looking successful. Batcher inspects each item and raises a `BackendError` naming how many failed and what the first one said, so a partial write is a failure rather than a silence.

Refresh is off by default, as it is in Elasticsearch itself. Forcing a refresh per batch is the standard way to make a bulk load an order of magnitude slower. Pass `refresh=True` when the write must be searchable before the call returns, which is usually only true in a test.

The key column stays in the document body. It is used as the `_id` *and* written as a field, because dropping it would make a read-write round trip lose it.

`overwrite` is refused past the first shard of a distributed write: every shard would empty the index, so each would discard the shards before it. Distribute an `upsert` instead.

## Requirements and limitations

The Arrow response for ES|QL, `format="arrow"`, needs Elasticsearch 8.18 or later. Against an older cluster the ES|QL call fails rather than degrading, so drop `esql=` and take the scroll path there.

The scroll path infers its schema from the first hit's `_source`. Heterogeneous documents, from a mapping that changed mid-index or an `object` field that is sometimes a scalar, produce a schema that does not describe the rest of the index. Constrain the shape with `esql=... | KEEP ...` or a projection so you read known columns.

Scroll contexts are a cluster resource. Every slice holds one open for a 2-minute window, refreshed as the engine drains it. Batcher clears them on the way out, best-effort, but a job killed mid-read leaves them to expire. Many concurrent sliced reads against a busy cluster can hurt production search latency.

The scroll path reads `_source`, so a field that is indexed but not stored in `_source` does not come back. ES|QL reads doc values and does not have this gap.

The scroll path answers `count()` exactly from the `_count` API in one round trip. An ES|QL result has no such endpoint, so count inside the pipeline with `| STATS COUNT(*)` and let the cluster compute it.

Batcher does not manage the index. Create it with the mappings, shard count and analyzers you want before the first write. An index Elasticsearch auto-creates from a bulk write gets dynamic mappings, which is rarely what a search index should have.

## See also

- {doc}`Writing to a database </integrations/databases/writing>`: the same mode vocabulary across SQL and the operational stores.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: sources, splits, pushdown.
- {doc}`Anomaly detection </cookbook/analytics/inference/anomaly-detection>`: the analysis a log index is
  usually pulled into.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the {py:class}`Source <batcher.io.Source>`/{py:class}`Split <batcher.io.Split>` protocol, for a store not listed.
- {doc}`I/O API </api/relational/io>`: the full reader and writer reference.
- {doc}`MongoDB </integrations/databases/mongodb>`: the other document store, read *and* write.
