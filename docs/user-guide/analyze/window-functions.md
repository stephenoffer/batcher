# Window functions

This page covers window functions in Batcher: ranking, running totals, lag and lead, and every other computation that reads a row's neighbors without collapsing them the way {py:meth}`group_by <batcher.Dataset.group_by>` does. Spell a window as one {py:meth}`window(...) <batcher.Dataset.window>` call, or as an ordinary expression bound with `.over(...)`. Both lower to the same operator, and both scale across a cluster, including a global window with no `PARTITION BY`.

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

A window and a `group_by` can compute the same sum. The difference is what comes out: `group_by` returns one row per group and drops the columns it didn't aggregate, while a window returns every input row with the result added as a new column.

![The five rows of ds, category a with products x, y and z at prices 30, 10 and 20, and category b with products p and q at 40 and 15, take two paths. group_by("category").agg(total=bt.col("price").sum()) collapses them to two rows, a with total 60 and b with total 55, and product and price are gone. ds.window(partition_by=["category"], functions={"cat_total": ("sum", "price")}) keeps all five rows with their product and price and adds cat_total: 60 on each a row and 55 on each b row.](/_static/diagrams/window_vs_group_by.svg)

`window` takes four arguments:

- `partition_by`: the keys that split rows into independent windows.
- `order_by`: how rows are ordered within a partition. An entry is `"col"`, a `("col", descending_bool)` pair, or an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`.
- `functions`: a dict of output name to spec, covered in the sections below.
- `frame`: an optional `(start, end)` row frame, for aggregates.

## Ranking functions

Ranking specs are the bare strings `"row_number"`, `"rank"`, and `"dense_rank"`, and they require `order_by`.

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

`rank` leaves gaps after ties. `dense_rank` does not.

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

The *normalized* ranking specs are `"percent_rank"` and `"cume_dist"`, which are SQL `PERCENT_RANK` and `CUME_DIST`. {py:func}`percent_rank <batcher.percent_rank>` rescales each row's rank into `[0, 1]`, so the first row scores `0` and the last scores `1`. {py:func}`cume_dist <batcher.cume_dist>` gives the fraction of the partition at or below the current row. Either one expresses "the cheapest 10% within each category" without hard-coding a row count.

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

Quartiles and deciles come from {py:func}`ntile(n) <batcher.ntile>`, the SQL `NTILE`, which splits each ordered partition into `n` roughly equal buckets numbered `1..n`. It takes the bucket count as an argument, so a bare string won't do. Spell it with the top-level `ntile` constructor bound by {py:meth}`.over(...) <batcher.AggExpr.over>`, the form covered below:

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

An aggregate spec is a tuple `(func, column)` where `func` is one of `"sum"`, `"avg"`, `"min"`, `"max"`, or `"count"`. With no frame and no order, it covers the whole partition.

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

`frame=(start, end)` bounds an aggregate to a row range measured from the row being computed. A negative offset is preceding, `0` is that row itself, a positive offset is following, `None` is unbounded. A running total, then, is everything from the start of the partition up to here.

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

### Counting rows, peers, or values

A frame offset counts in *units*, and which units you pick changes the answer. Pass them as a third element of the frame tuple.

| Units | An offset of `n` covers | Reach for it when |
|---|---|---|
| `"rows"` (default) | `n` physical rows | The window is a fixed number of observations. |
| `"groups"` | `n` peer groups, meaning distinct ORDER BY values | Ties should count once, not once per row. |
| `"range"` | Rows whose ORDER BY value is within `n` | The window is a span of time or of any measured quantity. |

The units part ways at a tie in the ORDER BY key. A frame that ends at the current row stops at that row under `"rows"`, but under `"range"` it runs to the last row of the current row's peer group, so every tied row gets the same answer.

![Three panels. First, the rows are sorted once by the PARTITION BY keys and then the ORDER BY keys, so every partition comes out contiguous, and each function's output column is scattered back to the row it came from in the original row order. Second, one ordered partition with ORDER BY values 1, 2, 2, 2, 5, 5, 7 and 9, where the current row is the middle of the three rows tied at 2, a peer group. A ROWS frame from the start of the partition to the current row stops at that row, so each tied row gets a different answer. A RANGE frame with the same bounds runs to the end of the peer group, so all three tied rows get the same answer. Third, GROUPS counts peer groups the way ROWS counts rows. A numeric RANGE offset is a binary search over the key's values rather than a walk over peers, and a null order key frames only its own null peer group. Both frame edges only slide right, so a frame is a FIFO queue and no frame is rescanned.](/_static/diagrams/window_frame_eval.svg)

The `"range"` unit is what a time series usually wants. A `"rows"` frame of 10 means something different when a sensor reports twice a minute than when it reports two hundred times. A `"range"` frame of five minutes means five minutes either way.

Offsets are in the ORDER BY key's own units, and microseconds for any timestamp or date key, whatever resolution it is stored at. A `"range"` offset needs exactly one ORDER BY key, and a numeric or temporal one, because the bound is arithmetic on it.

```python
import datetime as dt

base = dt.datetime(2024, 1, 1, 9, 0)
readings = bt.from_pydict(
    {
        "at": [base, base + dt.timedelta(minutes=1), base + dt.timedelta(minutes=30)],
        "reading": [1.0, 3.0, 5.0],
    }
)
five_minutes = 5 * 60 * 1_000_000
print(
    readings.with_columns(
        recent=bt.col("reading").sum().over(order_by=["at"], frame=(-five_minutes, 0, "range"))
    ).to_pydict()["recent"]
)
# [1.0, 4.0, 5.0]
```

The third reading is half an hour later, so its window holds only itself. Spelling that window out in microseconds is precise but not pleasant, so the `rolling_*_by` family takes the duration directly:

```python
print(
    readings.with_columns(
        recent=bt.col("reading").rolling_sum_by("at", "5m"),
        seen=bt.col("reading").rolling_count_by("at", "5m"),
    ).to_pydict()
)
# {'at': [datetime.datetime(2024, 1, 1, 9, 0), datetime.datetime(2024, 1, 1, 9, 1),
#         datetime.datetime(2024, 1, 1, 9, 30)],
#  'reading': [1.0, 3.0, 5.0], 'recent': [1.0, 4.0, 5.0], 'seen': [1, 2, 1]}
```

`rolling_count_by` over the same window is worth pairing with the average it accompanies. It says how much data the average was computed from, which is the difference between a quiet period and a broken sensor. Both endpoints are included, so a row exactly `window_size` back is in the window. That is Polars' `closed="both"`, and the SQL `RANGE BETWEEN ... PRECEDING AND CURRENT ROW` these lower to.

## Value functions

Value specs are `(func, column)` for `"first_value"` and `"last_value"`, `(func, column, offset)` for `"lag"` and `"lead"`, and `(func, column, n)` for `"nth_value"`, which reads the `n`-th row of the frame. {py:func}`nth_value <batcher.nth_value>` is SQL `NTH_VALUE`, and {py:func}`first_value <batcher.first_value>` is its special case `n = 1`. Use `nth_value` when the reference point is a fixed rank, such as "each product's price relative to its category's second-cheapest". With an `order_by` and no `frame`, `last_value` and `nth_value` use SQL's default frame, which ends at the current row, so `nth_value` stays null until the partition reaches its `n`-th row. Pass `frame=(None, None)` to read the whole partition on every row. The `first_value`, `last_value` and `nth_value` constructors also take `ignore_nulls=True`, which is SQL's `IGNORE NULLS`.

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
#  'top': [10, 10, 10, 15, 15], 'second': [None, 20, 20, None, 40]}
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

## Windows inside ordinary expressions

A window expression is an ordinary expression. Combine it with arithmetic, with a comparison, or with a second window, inside `select`, `with_columns`, or `filter`. The engine lifts each window into its own `Window` operator and rewrites the surrounding expression to read the result, exactly as a SQL engine does for `x - lag(x) OVER (...)`.

```python
prices = bt.from_pydict({"category": ["a", "a", "b", "b"], "price": [10, 20, 40, 15]})

shares = prices.with_columns(
    share=bt.col("price") / bt.col("price").sum().over(partition_by=["category"])
)
print(shares.to_pydict())
# {'category': ['a', 'a', 'b', 'b'], 'price': [10, 20, 40, 15],
#  'share': [0.3333333333333333, 0.6666666666666666, 0.7272727272727273, 0.2727272727272727]}
```

The window sees every input row before the filter runs. So a window in a predicate says "rows above their group's mean" outright, with none of the subquery SQL needs:

```python
above = prices.filter(bt.col("price") > bt.col("price").mean().over(partition_by=["category"]))
print(above.to_pydict())
# {'category': ['a', 'b'], 'price': [20, 40]}
```

Windows may not appear where SQL also forbids them, meaning inside `group_by().agg(...)`, in a join key, or in a sort key. Compute the window in a {py:meth}`with_columns <batcher.Dataset.with_columns>` step first, then reference the resulting column.

## Expression shorthands

Common window shapes have named methods on `Expr`, so you rarely spell the window out. They all accept `partition_by` / `order_by` and lower to the windows above.

The ones that depend on row order, such as `diff`, `pct_change`, `shift` and `cum_sum`, require an order. Batcher keeps no arrival order across a parallel scan, so it raises a `PlanError` rather than guess one. Pass `order_by=`, or bind the expression with `.over(order_by=...)`. When the data has no ordering column, add `.with_row_index("_row")` right after reading and order by `"_row"`.

The ones that depend on row order, such as `diff`, `pct_change`, `shift` and `cum_sum`, require an order. Batcher keeps no arrival order across a parallel scan, so it raises a `PlanError` rather than guess one. Pass `order_by=`, or bind the expression with `.over(order_by=...)`. When the data has no ordering column, add `.with_row_index("_row")` right after reading and order by `"_row"`.

```python
ts = bt.from_pydict({"day": [1, 2, 3], "price": [10, 15, 30]})
print(
    ts.with_columns(
        change=bt.col("price").diff(order_by="day"),  # price - lag(price)
        growth=bt.col("price").pct_change(order_by="day"),  # price / lag(price) - 1
        rnk=bt.col("price").rank(),  # RANK() OVER (ORDER BY price)
    ).to_pydict()
)
# {'day': [1, 2, 3], 'price': [10, 15, 30], 'change': [None, 5, 15],
#  'growth': [None, 0.5, 1.0], 'rnk': [1, 2, 3]}
```

`col("x").is_duplicated()` and `col("x").is_unique()` are the same idea, a `count(1) OVER (PARTITION BY x)` compared against 1. Both are most useful inside `filter`.

## Series recurrences

A frame answers each row from a bounded set of neighbors. Some time-series questions can't be asked that way, because the answer depends on the whole ordered prefix through a recurrence. Those have their own functions: `forward_fill`, `interpolate`, `rle_id`, and the exponentially weighted `ewm_mean`, `ewm_std`, and `ewm_var`. All of them require `order_by`, and all of them restart at every partition, so one device's readings never leak into another's.

`ewm_mean` weights each reading by `(1-alpha)^age`, so recent values dominate and old ones fade instead of dropping off the cliff a fixed window has. `ewm_std` and `ewm_var` give the matching spread, which is what makes a live volatility band or control limit.

```python
noisy = bt.from_pydict({"t": [1, 2, 3, 4], "v": [10.0, 30.0, 12.0, 28.0]})
print(
    noisy.with_columns(
        smooth=bt.col("v").ewm_mean(span=3).over(order_by=["t"]),
        spread=bt.col("v").ewm_std(span=3).over(order_by=["t"]),
    ).to_pydict()
)
# {'t': [1, 2, 3, 4], 'v': [10.0, 30.0, 12.0, 28.0],
#  'smooth': [10.0, 23.333333333333336, 16.857142857142858, 22.8],
#  'spread': [None, 14.14213562373095, 11.032419757890185, 10.091014390464986]}
```

{doc}`/user-guide/analyze/time-series` works through the fills, interpolation, time-decayed smoothing, and run segmentation on an irregular sensor feed.

## How a window scales across a cluster

A window is a pipeline breaker, so how it distributes depends on where its work can be cut. Batcher finds one of two seams.

The usual seam is the key. `PARTITION BY` makes every partition independent, so the rows hash-shuffle by the partition keys and each partition is computed whole on one worker. The union of the workers' output is the single-node answer, which is why this shape scales with the cluster for any window function and any frame.

A window with no `PARTITION BY` has one partition over every row, so there is no key to shuffle on. Batcher cuts along the *order* instead. It range-partitions the rows by the leading `ORDER BY` column into buckets that are ordered relative to each other, computes the window on each bucket in parallel, then shifts each bucket's result by what the earlier buckets contributed. A `row_number` shifts by the rows before it, a `dense_rank` by the distinct keys before it, a running `sum` by the running total before it. Equal keys always land in one bucket, so no peer group straddles a cut, and any further `ORDER BY` keys are evaluated inside each bucket.

Most functions have such a shift:

| How the global value is recovered | Functions |
|---|---|
| A running offset from earlier buckets | `row_number`, `rank`, `dense_rank`, running `sum`, `count`, `min`, `max`, `avg`, `var`, `stddev`, the bitwise and boolean folds, `first_value` |
| The few rows just before the bucket | `lag` |
| Closed out once every bucket has run | `percent_rank`, `cume_dist`, `ntile`, `last_value` |

`lead`, `median`, a distinct count, the fills, and the EWM family have none, because each reads rows its bucket does not hold in a direction no bounded exchange recovers. A window with no `PARTITION BY`, no `ORDER BY` and an aggregate is simpler still: every row gets the same value, so it runs as an ordinary distributed aggregate and broadcasts the scalar back.

## Requirements and limitations

- A global window distributes only when its leading `ORDER BY` key is a plain column of a type the range partitioner can cut. An expression key such as `order_by=[bt.col("a") + bt.col("b")]` has no distributed path.
- An explicit frame on a global window has no distributed path. Frames on a `PARTITION BY` window are unaffected.
- A top-N filter over a global ranking window, such as `.with_columns(r=bt.row_number().over(order_by="t")).filter(bt.col("r") <= 100)`, has no distributed path either. The filter fuses into the window as a rank bound, and a bucket knows only the rank within itself.
- `row_number()` over an `ORDER BY` key with duplicate values gives tied rows an arbitrary order, so which of them gets which number can differ between a single-node and a distributed run. This is true of any window, partitioned or not. Order by a unique key, or add a tiebreaker column, when the exact numbers matter.
- When a shape has no distributed path, `collect(distributed=True)` raises and names the functions at fault, rather than quietly running the whole relation on one node. Add a `PARTITION BY`, or pass `distributed=False` to run that stage on one node explicitly.
- Out of core, a `PARTITION BY` window spills by grace-partitioning on its partition keys. A global window with running offsets streams bucket by bucket, so peak memory is one bucket rather than the whole relation. The functions closed out after the last bucket need every bucket assembled first, so they don't stream this way.

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: collapse groups into summary rows.
- {doc}`Joins </user-guide/analyze/joins>`: combine windowed output with other datasets.
- {doc}`Expressions API </api/relational/expressions>`: the reference for every window, ranking and rolling method.
- {doc}`Time series </user-guide/analyze/time-series>`: bucketing, gap filling, and smoothing built from these windows.
- {doc}`Sorting </user-guide/transform/rows/sorting>`: top-n without a window, when you want the rows rather than a rank.
- {doc}`/cookbook/expressions/scalar/window_functions`: windows as a runnable script.
