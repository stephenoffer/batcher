# Elasticsearch

**Elasticsearch is read-only in Batcher.** There is a source; there is no sink.
`bt.read.elasticsearch(...)` pulls an index into the engine. If you need to write *back* to
Elasticsearch, use the cluster's own bulk API from your application. A search index is a serving
system with its own mappings and refresh semantics, and a columnar batch writer has no business
pretending otherwise.

| | |
| --- | --- |
| **Read** | `bt.read.elasticsearch(hosts=..., index=..., esql=...)` |
| **Write** | Not supported. Use the cluster's own bulk API. |
| **Extra** | `pip install 'batcher-engine[elasticsearch]'` (ES 8.18+ for the Arrow path) |
| **Parallelism** | Sliced scroll: one split per slice. The ES\|QL path is a single split. |
| **Pushdown** | Predicates become an appended `\| WHERE` (ES\|QL) or a `bool` query (scroll) |
| **Credentials** | `hosts` and `api_key`, stored on the source and never logged |

## Two read paths, and they are not equivalent

::::{tab-set}

:::{tab-item} ES|QL with Arrow output

This is the one you want. Elasticsearch 8.18+ can return an ES|QL result as an Arrow stream, which
Batcher reads straight into `RecordBatch`es with no per-row Python. The cluster does the filtering
and the aggregation; you get columns back.

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
Omitting `esql=` is not a stylistic choice. It silently drops you onto the scroll path, with a
JSON document per hit crossing Python before it becomes a column.
:::

## Credentials

`hosts` and `api_key` are stored verbatim on the source and never logged. The connector's identity
is the index name alone. Use an API key scoped to the indices you read. A scroll holds a cursor open
on the cluster, and you don't want that key to be able to do anything else.

## Predicate pushdown

Kyber pushes the query's filter into the cluster, and how depends on the path. On the ES|QL path it
becomes an appended `| WHERE` clause; on the scroll path it becomes an ES `bool` query AND-merged
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

The ES|QL path does **not** slice. The whole result comes back in one Arrow stream, as a single
split. That is the tradeoff, and it is usually the right one: ES|QL pushes the filter and the
projection into the cluster, so what crosses the wire is small, and one fast stream of a small
result beats eight slow scrolls of a large one. Reach for slices when you genuinely need to pull a
large, unaggregated slab of documents out.

:::{tip}
Sizing slices above the index's shard count buys nothing, because a slice cannot span a shard.
Match `segments` to shards, not to your CPU count.
:::

## Failure modes worth knowing

**No Arrow on an older cluster.** The `format="arrow"` ES|QL response is 8.18+. Against an older
cluster the ES|QL call fails rather than silently degrading. Drop `esql=` and take the scroll path.

**Schema comes from one document.** Scrolling infers the Arrow schema from the first hit's
`_source`. Documents with heterogeneous fields, from a mapping that changed mid-index or an
`object` field that is sometimes a scalar, produce a schema that does not describe the rest of the
index. Constrain the shape with `esql=... | KEEP ...` (or a projection) so you are reading known
columns rather than a union of everything anyone ever indexed.

:::{warning}
**Scroll contexts are a cluster resource.** Every slice holds one alive for its 2-minute window,
refreshed as the engine drains it. Batcher clears them on the way out, best-effort, but a job
killed mid-read leaves them to expire on their own. Many concurrent sliced reads against a busy
cluster is a way to hurt production search latency.
:::

**`_source` only.** Scrolling reads `_source`, so a field that is indexed but not stored in
`_source` does not come back. ES|QL, which reads doc values, does not have this problem. One more
reason to prefer it.

**No row count.** There is no cheap exact count, so `count()` reads. If you want a count, ask ES|QL
for one (`| STATS COUNT(*)`) and let the cluster compute it.

## See also

- [Reading data](../user-guide/reading-data.md): sources, splits, pushdown.
- [Anomaly detection](../examples/analytics/anomaly-detection.md): the analysis a log index is
  usually pulled into.
- [Custom connectors](../user-guide/custom-connectors.md): the `Source`/`Split` protocol, if you
  need the sink this connector does not have.
- [I/O API](../api/io.md): the full reader reference.
- [MongoDB](mongodb.md): the other document store, read *and* write.
