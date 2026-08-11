# Top k per group

The two best-selling products in every category. Not the two best-selling products
overall, but the two best in *each* category. Different query, and a much easier one to get
wrong.

## The data

Nine products, three categories. Two deliberate ties: `b1` and `b2` both did 500, and
`t2` and `t3` both did 400.

```python
import batcher as bt
from batcher import col, rank, row_number

sales = bt.from_pydict(
    {
        "category": ["books", "books", "books", "books", "toys", "toys", "toys", "games", "games"],
        "product": ["b1", "b2", "b3", "b4", "t1", "t2", "t3", "g1", "g2"],
        "revenue": [500.0, 500.0, 300.0, 100.0, 900.0, 400.0, 400.0, 250.0, 700.0],
    }
)
print(sales.count())
# 9
```

## The trap

:::{warning}
`sort` then `limit` is a *global* top-k. It answers a question nobody asked: two rows, two
categories, and `books` missing from the answer entirely.
:::

```python
print(sales.top_k(2, "revenue").to_pydict())
# {'category': ['toys', 'games'], 'product': ['t1', 'g2'], 'revenue': [900.0, 700.0]}
```

`top_k` is sugar for `sort(...).limit(k)` and the engine fuses the pair into a heap, so it
is genuinely fast. It is fast at the wrong thing.

The other wrong turn is to loop: get the distinct categories, then run one query per
category with a `filter` and a `limit`. Now you have N queries, N scans of the table, and
a runtime linear in the number of categories. With three categories nobody notices. With
fifty thousand SKUs it is the whole afternoon.

## Rank inside the partition, then filter

One window, one filter, one scan. {py:func}`row_number() <batcher.row_number>` numbers the rows within each partition
in the order you give it, and keeping `rn <= k` keeps the top k of every group at once.

The SQL tab needs a subquery, because a window cannot appear in a `WHERE` clause: the
filter runs before the window does. That is the standard shape, and it is why every
top-k-per-group answer on the internet has a nested `SELECT`.

::::{tab-set}
:::{tab-item} DataFrame
```python
top2 = (
    sales.with_columns(
        rn=row_number().over(
            partition_by=["category"],
            order_by=[("revenue", True), ("product", False)],
        )
    )
    .filter(col("rn") <= 2)
    .select("category", "product", "revenue", "rn")
    .sort("category", "rn")
)
print(top2.to_pydict())
# {'category': ['books', 'books', 'games', 'games', 'toys', 'toys'],
#  'product': ['b1', 'b2', 'g2', 'g1', 't1', 't2'],
#  'revenue': [500.0, 500.0, 700.0, 250.0, 900.0, 400.0], 'rn': [1, 2, 1, 2, 1, 2]}
```
:::

:::{tab-item} SQL
```python
sql_top2 = bt.sql(
    """
    SELECT category, product, revenue
    FROM (
      SELECT category, product, revenue,
             ROW_NUMBER() OVER (
               PARTITION BY category ORDER BY revenue DESC, product
             ) AS rn
      FROM sales
    )
    WHERE rn <= 2
    ORDER BY category, revenue DESC, product
    """,
    sales=sales,
)
print(sql_top2.to_pydict())
# {'category': ['books', 'books', 'games', 'games', 'toys', 'toys'],
#  'product': ['b1', 'b2', 'g2', 'g1', 't1', 't2'],
#  'revenue': [500.0, 500.0, 700.0, 250.0, 900.0, 400.0]}
```
:::
::::

Six rows: two per category, every category represented. The window sorts within each
partition, not across the whole relation, and it does it in the same pass.

:::{note}
Batcher's DataFrame API does let you put a window inside `filter`, because it lifts the
window into its own operator and rewrites the predicate to read the result. Both spellings
build the same plan.
:::

## About that second sort key

`order_by=[("revenue", True), ("product", False)]` sorts by revenue descending, then by
product name ascending. The second key is not decoration.

:::{important}
`b1` and `b2` both did 500. `row_number` must hand out 1 and 2, so without a tiebreaker it
picks one of them arbitrarily, and "arbitrarily" means the answer can change between runs,
between partition counts, and between single-node and distributed.
:::

A report that flips its top seller every Tuesday is a report nobody trusts. Add a
deterministic tiebreaker and the question becomes well-posed.

## When the tie should not be broken

Sometimes you want *all* the leaders, ties and all. That is `rank`, which gives tied rows
the same number:

```python
tied = (
    sales.with_columns(
        rk=rank().over(partition_by=["category"], order_by=[("revenue", True)])
    )
    .filter(col("rk") <= 2)
    .select("category", "product", "revenue", "rk")
    .sort("category", "rk", "product")
)
print(tied.to_pydict())
# {'category': ['books', 'books', 'games', 'games', 'toys', 'toys', 'toys'],
#  'product': ['b1', 'b2', 'g2', 'g1', 't1', 't2', 't3'],
#  'revenue': [500.0, 500.0, 700.0, 250.0, 900.0, 400.0, 400.0],
#  'rk': [1, 1, 1, 2, 1, 2, 2]}
```

Seven rows, because `toys` has a two-way tie for second and both members come back.
{py:func}`dense_rank <batcher.dense_rank>` is the third option: it also ties, but it does not leave gaps after a tie, so
`rk <= 2` means "the top two distinct revenue values" rather than "the top two positions".

Three functions, three different questions:

| Want | Function | `books` at k=2 |
| --- | --- | --- |
| Exactly k rows | `row_number()` | `b1`, `b2` (tie broken by the sort key) |
| Every row in the top k positions | `rank()` | `b1`, `b2` |
| Every row in the top k *values* | `dense_rank()` | `b1`, `b2`, `b3` |

## k = 1 does not need a window

:::{tip}
If you only want the single best row per group, `arg_max` gets it as a plain aggregate.
Being an aggregate makes it mergeable, so it runs in bounded memory and merges across
partitions without ever materializing a group. A window has to hold each partition to sort
it, while `arg_max` holds one row.
:::

```python
best = (
    sales.group_by("category")
    .agg(product=col("product").arg_max(col("revenue")), revenue=col("revenue").max())
    .sort("category")
)
print(best.to_pydict())
# {'category': ['books', 'games', 'toys'], 'product': ['b1', 'g2', 't1'],
#  'revenue': [500.0, 700.0, 900.0]}
```

Reach for the window when k > 1. Reach for `arg_max` when k = 1 and the groups are large.

## See also

- {doc}`Window functions </user-guide/analyze/window-functions>`: ranking, frames, `ntile`.
- {doc}`Basket analysis </cookbook/analytics/inference/basket-analysis>`: ranking pairs by lift instead of rows by revenue.
- {doc}`Cohort analysis </cookbook/analytics/behavior/cohort-analysis>`: the other use of `partition_by` without a sort,
  where the window labels rather than ranks.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: what `sort` costs, and when it spills.
- {doc}`Window internals </architecture/deep-dives/operators/window-internals>`: why the partitioned sort beats
  the global one.
- {doc}`Sort internals </architecture/deep-dives/operators/sort-internals>`: the heap that `top_k` fuses into.
- {doc}`Expressions API </api/relational/expressions>`: `row_number`, `rank`, `dense_rank`, `arg_max`.
