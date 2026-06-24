# Analytics query

A multi-step analytical query — aggregate, join, and window — over a small orders
table. The same pipeline runs unchanged on millions of rows; the optimizer (Kyber)
plans the joins and aggregates, and re-plans at pipeline breakers on measured row
counts.

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

A running total within each region, ordered by amount — an aggregate made windowed
with `.over(...)`.

```python
running = orders.with_columns(
    running=col("amt").sum().over(partition_by=["region"], order_by=["amt"])
).sort("region", "amt")
print(running.to_pydict()["running"])
# [20, 60, 10, 40, 90]
```

Ranking functions work the same way — `rank().over(partition_by=..., order_by=...)`
numbers rows within each partition.

```python
ranked = orders.with_columns(
    position=rank().over(partition_by=["region"], order_by=["amt"])
).sort("region", "amt")
print(ranked.to_pydict()["position"])
# [1, 2, 1, 2, 3]
```

## As SQL

The same query expressed in SQL — it builds the same plan and returns a lazy
`Dataset`, so you can mix the two freely.

```python
out = bt.sql(
    "SELECT region, SUM(amt) AS revenue FROM orders GROUP BY region ORDER BY revenue DESC",
    orders=orders,
)
print(out.to_pydict())
# {'region': ['W', 'E'], 'revenue': [90, 60]}
```
