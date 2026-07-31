# Analytics query

Aggregate, join, then window, over a small orders table. Five rows here, but the same
pipeline runs unchanged on millions: Kyber plans the joins and aggregates up front,
then re-plans at each pipeline breaker once it has measured what the row counts really
were. Nothing about the code changes.

```python
import batcher as bt
from batcher import col, rank

orders = bt.from_pydict(
    {
        "region": ["W", "E", "W", "E", "W"],
        "rep": ["a", "b", "a", "c", "a"],
        "amt": [10, 20, 30, 40, 50],
    }
)
```

## Aggregate

Revenue and order count per region, biggest first.

```python
revenue = (
    orders.group_by("region")
    .agg(revenue=col("amt").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(revenue.to_pydict())
# {'region': ['W', 'E'], 'revenue': [90, 60], 'orders': [3, 2]}
```

## Join

Enrich with a region dimension, then aggregate on the joined column.

```python
regions = bt.from_pydict({"region": ["W", "E"], "name": ["West", "East"]})
by_name = (
    orders.join(regions, on="region")
    .group_by("name")
    .agg(revenue=col("amt").sum())
    .sort("name")
)
print(by_name.to_pydict())
# {'name': ['East', 'West'], 'revenue': [60, 90]}
```

## Window

A running total within each region, ordered by amount. Any aggregate becomes a window
function once you hang `.over(...)` off it, with the partition and the ordering given as
keyword arguments rather than a separate window object to declare first.

```python
running = orders.with_columns(
    running=col("amt").sum().over(partition_by=["region"], order_by=["amt"])
).sort("region", "amt")
print(running.to_pydict()["running"])
# [20, 60, 10, 40, 90]
```

Ranking functions take the same shape: `rank().over(partition_by=..., order_by=...)`
numbers rows within each partition.

```python
ranked = orders.with_columns(
    position=rank().over(partition_by=["region"], order_by=["amt"])
).sort("region", "amt")
print(ranked.to_pydict()["position"])
# [1, 2, 1, 2, 3]
```

## The same query in SQL

SQL builds the identical plan and hands back a lazy `Dataset`, so the two spellings
mix freely.

```python
out = bt.sql(
    "SELECT region, SUM(amt) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC",
    orders=orders,
)
print(out.to_pydict())
# {'region': ['W', 'E'], 'revenue': [90, 60]}
```

## What to change first

Three edits turn this into a real query, and each one is a single line:

1. Swap `from_pydict` for {doc}`a reader </user-guide/moving-data/reading-data>`, such as
   `bt.read.parquet("s3://bucket/orders/")`. Nothing below it changes.
1. Add a `filter` before the `group_by`. The optimizer pushes it toward the scan, so a
   partitioned or statistics-carrying source skips files rather than reading them. Confirm
   it did with {doc}`ds.explain() </user-guide/operate/explain-plans>`.
1. End with a write instead of a print: `out.write.parquet(...)`. See
   {doc}`/user-guide/moving-data/writing-data`.

## See also

- {doc}`/cookbook/analytics/index`: the focused recipes for cohorts, funnels, sessions, and top-k, each
  with the trap that makes it harder than it looks.
- {doc}`/user-guide/analyze/aggregations` and {doc}`/user-guide/analyze/window-functions`: the two
  operators this page leans on, in full.
- {doc}`/user-guide/analyze/joins`: join types, and which side gets built.
- {doc}`/cookbook/data-engineering/etl-pipeline`: the same treatment for an ingest pipeline, ending in a written table.
- {doc}`/tutorials/optimizing-a-slow-query`: what to do when this shape meets real data.
