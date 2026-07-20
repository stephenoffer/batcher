# Basket analysis

Which products get bought together? The answer is a self-join, and the self-join is
exactly where this goes wrong: done carelessly it counts every pair twice, pairs every
item with itself, and generates a table you cannot afford.

## The data

Five orders, one row per item in the basket.

```python
import batcher as bt
from batcher import col, lit

baskets = bt.from_pydict(
    {
        "order_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
        "item": [
            "bread", "butter", "jam",
            "bread", "butter",
            "bread", "jam", "milk",
            "butter", "jam",
            "bread",
        ],
    }
)
n_orders = baskets.n_unique("order_id")
print(n_orders)
# 5
```

## Co-occurrence, and the guard that makes it right

:::{warning}
Join the table to itself on `order_id` and you get every *ordered* pair of items in every
basket, including `(bread, bread)` and both `(bread, jam)` and `(jam, bread)`. Without a
guard that is 15 rows instead of 5, every count doubled, and four meaningless self-pairs
sitting at the top of the chart.
:::

:::{tip}
The `item < item_b` filter fixes both problems at once: it drops the self-pairs, because
nothing is less than itself, and it keeps exactly one of each unordered pair.
:::

::::{tab-set}
:::{tab-item} DataFrame
```python
pairs = (
    baskets.join(baskets.rename({"item": "item_b"}), on="order_id")
    .filter(col("item") < col("item_b"))
    .group_by("item", "item_b")
    .agg(both=bt.count())
    .sort("both", descending=True)
)
print(pairs.to_pydict())
# {'item': ['bread', 'bread', 'butter', 'bread', 'jam'],
#  'item_b': ['jam', 'butter', 'jam', 'milk', 'milk'],
#  'both': [2, 2, 2, 1, 1]}
```
:::

:::{tab-item} SQL
```python
sql_pairs = bt.sql(
    """
    SELECT a.item AS item, b.item AS item_b, COUNT(*) AS both
    FROM baskets a
    INNER JOIN baskets b ON a.order_id = b.order_id
    WHERE a.item < b.item
    GROUP BY a.item, b.item
    ORDER BY both DESC, item, item_b
    """,
    baskets=baskets,
)
print(sql_pairs.to_pydict()["both"])
# [2, 2, 2, 1, 1]
```
:::
::::

The guard also halves the work, which matters more than the aesthetics. A basket with `k`
items produces `k * k` rows from the join and `k * (k - 1) / 2` after the filter. Batcher pushes the
predicate down, so the filter runs as the join emits rather than after it materializes.
The join itself is still quadratic *per basket*, though. One pathological order with 2,000
line items contributes four million rows on its own. Cap it before you join:

```python
sane = baskets.filter(
    col("order_id").count().over(partition_by=["order_id"]) <= 50
)
print(sane.count())
# 11
```

Nothing is dropped here, because no basket has 50 items. On real data that predicate is
the difference between a query that finishes and a query that does not.

## Support, confidence, lift

Counts alone are useless: `bread` co-occurs with everything because `bread` is in almost
every basket. Three standard measures fix that.

*Support* is the fraction of all orders containing both items, which answers "how common is
this pair?". *Confidence* is the fraction of the orders containing A that also contain B, and
it has a direction. *Lift* divides confidence by B's own base rate: how much more likely is B
given A than B in general? A lift of 1.0 means A tells you nothing about B.

```python
item_orders = baskets.group_by("item").agg(orders=col("order_id").n_unique())

rules = (
    pairs.join(item_orders.rename({"orders": "a_orders"}), on="item")
    .join(item_orders.rename({"item": "item_b", "orders": "b_orders"}), on="item_b")
    .with_columns(
        support=col("both") / lit(float(n_orders)),
        confidence=col("both") / col("a_orders"),
        lift=(col("both") * lit(float(n_orders))) / (col("a_orders") * col("b_orders")),
    )
    .sort("lift", descending=True)
)
print(rules.select("item", "item_b", "both", "support", "confidence", "lift").to_pydict())
# {'item': ['jam', 'bread', 'butter', 'bread', 'bread'],
#  'item_b': ['milk', 'milk', 'jam', 'jam', 'butter'],
#  'both': [1, 1, 2, 2, 2],
#  'support': [0.2, 0.2, 0.4, 0.4, 0.4],
#  'confidence': [0.3333333333333333, 0.25, 0.6666666666666666, 0.5, 0.5],
#  'lift': [1.6666666666666667, 1.25, 1.1111111111111112, 0.8333333333333334, 0.8333333333333334]}
```

Rank the same five pairs by raw count and by lift and you get two different leaderboards,
in opposite orders:

| Pair | Co-occurrences | Support | Lift |
| --- | --- | --- | --- |
| jam, milk | 1 | 0.2 | 1.67 |
| bread, milk | 1 | 0.2 | 1.25 |
| butter, jam | 2 | 0.4 | 1.11 |
| bread, jam | 2 | 0.4 | 0.83 |
| bread, butter | 2 | 0.4 | 0.83 |

Read the bottom row. `bread -> butter` has the joint-highest raw count (2) and the
joint-highest support (0.4), and its lift is **0.83**, below 1. Bread is in four of five
baskets, so butter appearing alongside it is not a signal, it is arithmetic. Rank by count
and you would ship "customers who buy bread also buy butter" as an insight. Rank by lift
and `jam -> milk` comes out on top, on a single co-occurrence.

:::{important}
Which is the other half of the lesson: lift is loud on rare pairs. One order is not
evidence. Filter on support *before* you rank on lift (`support >= 0.01` on a real
catalogue), and treat anything below that as noise no matter how good the lift looks.
:::

## Confidence has a direction

`item < item_b` keeps one row per unordered pair. That is right for support and lift, which
are symmetric, and wrong for confidence, which is not. `confidence(bread -> jam)` is 2/4 = 0.5.
`confidence(jam -> bread)` is 2/3 = 0.67. The table above only has the first.

If you want both directions, swap the guard for `!=`, which keeps both orderings and still
excludes self-pairs. It doubles the row count, and it is the right call when you are
generating recommendation rules rather than a symmetric co-occurrence matrix.

```python
directed = (
    baskets.join(baskets.rename({"item": "item_b"}), on="order_id")
    .filter(col("item") != col("item_b"))
    .group_by("item", "item_b")
    .agg(both=bt.count())
)
print(directed.count())
# 10
```

Ten rows: five unordered pairs, each in both directions.

## See also

:::{seealso}
- [Top k per group](top-k-per-group.md): keep the best three partners per item rather than
  the global leaderboard.
- [Geospatial binning](geospatial-binning.md): the other recipe whose whole difficulty is
  choosing a key that groups.
- [Joins](../../user-guide/joins.md): the join engine, and what a self-join costs.
- [Distinct and dedup](../../user-guide/distinct-and-dedup.md): duplicate line items in a
  basket will inflate every count on this page, so dedupe `(order_id, item)` first.
- [Join algorithms](../../deep-dives/join-algorithms.md): how the build side and the
  pushed-down predicate keep the quadratic bounded.
- [Expressions API](../../api/expressions.md): `lit`, `n_unique`, `count().over(...)`.
:::
