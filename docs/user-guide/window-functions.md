# Window functions

A window function computes a value for each row from a set of related rows, without
collapsing them the way `group_by` does. Call `window(...)` with the partition keys
and a dict that maps each output column name to a function spec. Most specs also
want an ordering.

## The window call

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "a", "a", "b", "b"],
        "product": ["x", "y", "z", "p", "q"],
        "price": [30, 10, 20, 40, 15],
    }
)
```

`window` takes:

- `partition_by`: the keys that split rows into independent windows.
- `order_by`: how rows are ordered within a partition. An entry is `"col"`, a
  `("col", descending_bool)` pair, or an `Expr`.
- `functions`: a dict of output name to spec (see below).
- `frame`: an optional `(start, end)` row frame, for aggregates.

## Ranking functions

Ranking specs are the bare strings `"row_number"`, `"rank"`, and `"dense_rank"`,
and they require `order_by`.

```python
ranked = ds.window(
    partition_by=["category"],
    order_by=[("price", True)],
    functions={"rnk": "row_number"},
).sort("category", "rnk")
print(ranked.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['x', 'z', 'y', 'p', 'q'],
#  'price': [30, 20, 10, 40, 15], 'rnk': [1, 2, 3, 1, 2]}
```

`rank` leaves gaps after ties; `dense_rank` does not.

```python
ranks = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"rk": "rank", "dr": "dense_rank"},
).sort("category", "price")
print(ranks.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['y', 'z', 'x', 'q', 'p'],
#  'price': [10, 20, 30, 15, 40], 'rk': [1, 2, 3, 1, 2], 'dr': [1, 2, 3, 1, 2]}
```

The *normalized* ranking specs are `"percent_rank"` and `"cume_dist"` (SQL
`PERCENT_RANK` / `CUME_DIST`). `percent_rank` rescales each row's rank into
`[0, 1]`, giving `0` to the first row and `1` to the last; `cume_dist` gives the
fraction of the partition at or below the current row. Either one expresses "the
cheapest 10% within each category" without hard-coding a row count.

```python
norm = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"pr": "percent_rank", "cd": "cume_dist"},
).sort("category", "price")
print(norm.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['y', 'z', 'x', 'q', 'p'],
#  'price': [10, 20, 30, 15, 40], 'pr': [0.0, 0.5, 1.0, 0.0, 1.0],
#  'cd': [0.3333333333333333, 0.6666666666666666, 1.0, 0.5, 1.0]}
```

Quartiles and deciles come from `ntile(n)`, which splits each ordered partition
into `n` roughly equal buckets numbered `1..n` (SQL `NTILE`). It takes the bucket
count as an argument, so a bare string won't do; spell it with the top-level
`ntile` constructor bound by `.over(...)`, the form covered below:

```python
from batcher import ntile

quartiles = ds.with_columns(
    bucket=ntile(2).over(partition_by=["category"], order_by=["price"])
).sort("category", "price")
print(quartiles.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['y', 'z', 'x', 'q', 'p'],
#  'price': [10, 20, 30, 15, 40], 'bucket': [1, 1, 2, 1, 2]}
```

## Aggregate functions

An aggregate spec is a tuple `(func, column)` where `func` is one of `"sum"`,
`"avg"`, `"min"`, `"max"`, or `"count"`. With no frame and no order, it covers the
whole partition.

```python
totals = ds.window(
    partition_by=["category"],
    functions={"cat_total": ("sum", "price")},
).sort("category", "product")
print(totals.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['x', 'y', 'z', 'p', 'q'],
#  'price': [30, 10, 20, 40, 15], 'cat_total': [60, 60, 60, 55, 55]}
```

## Frames

`frame=(start, end)` bounds an aggregate to a row range measured from the row being
computed. A negative offset is preceding, `0` is that row itself, a positive offset
is following, `None` is unbounded. A running total, then, is everything from the
start of the partition up to here.

```python
running = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"running": ("sum", "price")},
    frame=(None, 0),
).sort("category", "price")
print(running.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['y', 'z', 'x', 'q', 'p'],
#  'price': [10, 20, 30, 15, 40], 'running': [10, 30, 60, 15, 55]}
```

## Value functions

Value specs are `(func, column)` for `"first_value"` and `"last_value"`,
`(func, column, offset)` for `"lag"` and `"lead"`, and `(func, column, n)` for
`"nth_value"`, which reads the `n`-th row of the ordered partition (SQL
`NTH_VALUE`; `first_value` is the special case `n = 1`). Use `nth_value` when the
reference point is a fixed rank: "each product's price relative to its category's
second-cheapest".

```python
shifted = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={
        "prev": ("lag", "price", 1),
        "top": ("first_value", "price"),
        "second": ("nth_value", "price", 2),
    },
).sort("category", "price")
print(shifted.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'product': ['y', 'z', 'x', 'q', 'p'],
#  'price': [10, 20, 30, 15, 40], 'prev': [None, 10, 20, None, 15],
#  'top': [10, 10, 10, 15, 15], 'second': [20, 20, 20, 40, 40]}
```

## Top-N per partition

Ranking plus a filter gives the top rows per group.

```python
top1 = (
    ds.window(
        partition_by=["category"],
        order_by=[("price", True)],
        functions={"rnk": "row_number"},
    )
    .filter(bt.col("rnk") == 1)
    .select("category", "product", "price")
)
print(top1.to_pydict())
# {'category': ['a', 'b'], 'product': ['x', 'p'], 'price': [30, 40]}
```

## Composing a window with ordinary expressions

A window expression is an ordinary expression. Combine it with arithmetic, with a
comparison, or with a second window, inside `select`, `with_columns`, or `filter`.
The engine lifts each window into its own `Window` operator and rewrites the
surrounding expression to read the result, exactly as a SQL engine does for
`x - lag(x) OVER (...)`.

```python
prices = bt.from_pydict({"category": ["a", "a", "b", "b"], "price": [10, 20, 40, 15]})

shares = prices.with_columns(
    share=bt.col("price") / bt.col("price").sum().over(partition_by=["category"])
)
print(shares.to_pydict())
# {'category': ['a', 'a', 'b', 'b'], 'price': [10, 20, 40, 15],
#  'share': [0.3333333333333333, 0.6666666666666666, 0.7272727272727273, 0.2727272727272727]}
```

The window sees every input row before the filter runs. So a window in a predicate
says "rows above their group's mean" outright, with none of the subquery SQL needs:

```python
above = prices.filter(bt.col("price") > bt.col("price").mean().over(partition_by=["category"]))
print(above.to_pydict())
# {'category': ['a', 'b'], 'price': [20, 40]}
```

Windows may not appear where SQL also forbids them: inside `group_by().agg(...)`,
in a join key, in a sort key. Compute the window in a `with_columns` step first,
then reference the resulting column.

## Expression shorthands

Common window shapes have named methods on `Expr`, so you rarely spell the window
out. They all accept `partition_by` / `order_by` and lower to the windows above.

```python
ts = bt.from_pydict({"price": [10, 15, 30]})
print(
    ts.with_columns(
        change=bt.col("price").diff(),          # price - lag(price)
        growth=bt.col("price").pct_change(),    # price / lag(price) - 1
        rnk=bt.col("price").rank(),             # RANK() OVER (ORDER BY price)
    ).to_pydict()
)
# {'price': [10, 15, 30], 'change': [None, 5, 15],
#  'growth': [None, 0.5, 1.0], 'rnk': [1, 2, 3]}
```

`col("x").is_duplicated()` and `col("x").is_unique()` are the same idea (a
`count(1) OVER (PARTITION BY x)` compared against 1). Both are most useful inside
`filter`.

## Next steps

- [Aggregations](aggregations.md): collapse groups into summary rows.
- [Joins](joins.md): combine windowed output with other datasets.
- [Expressions API](../api/expressions.md): the reference for every window, ranking
  and rolling method.
