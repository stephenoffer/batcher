# Window functions

Window functions compute a value for each row from a set of related rows, without
collapsing the rows the way `group_by` does. Call `window(...)` with the partition
keys, an order, and a dictionary mapping each output column name to a function
spec.

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

- `partition_by` - the keys that split rows into independent windows.
- `order_by` - ordering within each partition; entries are `"col"`,
  `("col", descending_bool)`, or an `Expr`.
- `functions` - a dict of output name to spec (see below).
- `frame` - an optional `(start, end)` row frame for aggregate functions.

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
# {'category': ['a', 'a', 'a', 'b', 'b'], 'price': [30, 20, 10, 40, 15],
#  'product': ['x', 'z', 'y', 'p', 'q'], 'rnk': [1, 2, 3, 1, 2]}
```

`rank` leaves gaps after ties; `dense_rank` does not.

```python
ranks = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"rk": "rank", "dr": "dense_rank"},
).sort("category", "price")
print(ranks.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'price': [10, 20, 30, 15, 40],
#  'product': ['y', 'z', 'x', 'q', 'p'], 'rk': [1, 2, 3, 1, 2], 'dr': [1, 2, 3, 1, 2]}
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
# {'category': ['a', 'a', 'a', 'b', 'b'], 'price': [30, 10, 20, 40, 15],
#  'product': ['x', 'y', 'z', 'p', 'q'], 'cat_total': [60, 60, 60, 55, 55]}
```

## Frames

`frame=(start, end)` bounds an aggregate to a row range relative to the current
row: a negative offset is preceding, `0` is the current row, a positive offset is
following, and `None` is unbounded. A running total is "everything from the start
of the partition through the current row".

```python
running = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"running": ("sum", "price")},
    frame=(None, 0),
).sort("category", "price")
print(running.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'price': [10, 20, 30, 15, 40],
#  'product': ['y', 'z', 'x', 'q', 'p'], 'running': [10, 30, 60, 15, 55]}
```

## Value functions

Value specs are `(func, column)` for `"first_value"` and `"last_value"`, and
`(func, column, offset)` for `"lag"` and `"lead"`.

```python
shifted = ds.window(
    partition_by=["category"],
    order_by=[("price", False)],
    functions={"prev": ("lag", "price", 1), "top": ("first_value", "price")},
).sort("category", "price")
print(shifted.to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b'], 'price': [10, 20, 30, 15, 40],
#  'product': ['y', 'z', 'x', 'q', 'p'], 'prev': [None, 10, 20, None, 15],
#  'top': [10, 10, 10, 15, 15]}
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

A window expression is an ordinary expression: it can be combined with arithmetic,
comparisons, and other windows inside `select`, `with_columns`, and `filter`. The
engine lifts each window into its own `Window` operator and rewrites the surrounding
expression to read the result, exactly as a SQL engine does for
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

Because the window sees every input row before the filter runs, a window in a
predicate expresses "rows above their group's mean" directly — the subquery SQL
would need:

```python
above = prices.filter(bt.col("price") > bt.col("price").mean().over(partition_by=["category"]))
print(above.to_pydict())
# {'category': ['a', 'b'], 'price': [20, 40]}
```

Windows may not appear where SQL also forbids them — inside `group_by().agg(...)`,
a join key, or a sort key. Compute the window in a `with_columns` step first and
reference the resulting column.

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

`col("x").is_duplicated()` and `col("x").is_unique()` are the same idea — a
`count(1) OVER (PARTITION BY x)` compared against 1 — and are most useful inside
`filter`.

## Next steps

- [Aggregations](aggregations.md): collapse groups into summary rows.
- [Joins](joins.md): combine windowed output with other datasets.
