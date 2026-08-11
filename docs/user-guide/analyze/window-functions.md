# Window functions

A window function computes a value for each row from a set of related rows, without
collapsing them the way {py:meth}`group_by <batcher.Dataset.group_by>` does. Call {py:meth}`window(...) <batcher.Dataset.window>` with the partition keys
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
  `("col", descending_bool)` pair, or an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`.
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

The *normalized* ranking specs are `"percent_rank"` and `"cume_dist"`, which are SQL
`PERCENT_RANK` and `CUME_DIST`. {py:func}`percent_rank <batcher.percent_rank>` rescales each row's rank into `[0, 1]`,
giving `0` to the first row and `1` to the last. {py:func}`cume_dist <batcher.cume_dist>` gives the fraction of the
partition at or below the current row. Either one expresses "the cheapest 10% within each
category" without hard-coding a row count.

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

Quartiles and deciles come from {py:func}`ntile(n) <batcher.ntile>`, the SQL `NTILE`, which splits each ordered
partition into `n` roughly equal buckets numbered `1..n`. It takes the bucket count as an
argument, so a bare string won't do. Spell it with the top-level `ntile` constructor
bound by {py:meth}`.over(...) <batcher.AggExpr.over>`, the form covered below:

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

### Counting rows, peers, or values

A frame offset counts in *units*, and which units you pick changes the answer. Pass them as
a third element of the frame tuple.

| Units | An offset of `n` covers | Reach for it when |
|---|---|---|
| `"rows"` (default) | `n` physical rows | The window is a fixed number of observations. |
| `"groups"` | `n` peer groups, meaning distinct ORDER BY values | Ties should count once, not once per row. |
| `"range"` | Rows whose ORDER BY **value** is within `n` | The window is a span of time or of any measured quantity. |

The last one is what a time series usually wants. A `"rows"` frame of 10 means something
different when a sensor reports twice a minute than when it reports two hundred times; a
`"range"` frame of five minutes means five minutes either way.

Offsets are in the ORDER BY key's own units, and **microseconds for any timestamp or date
key**, whatever resolution it is stored at. A `"range"` offset needs exactly one ORDER BY
key, and a numeric or temporal one, because the bound is arithmetic on it.

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

The third reading is half an hour later, so its window holds only itself. Spelling that
window out in microseconds is precise but not pleasant, so the `rolling_*_by` family takes
the duration directly:

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

`rolling_count_by` over the same window is worth pairing with the average it accompanies: it
says how much data the average was computed from, which is the difference between a quiet
period and a broken sensor. Both endpoints are included, so a row exactly `window_size` back
is in the window — Polars' `closed="both"`, and the SQL `RANGE BETWEEN … PRECEDING AND
CURRENT ROW` these lower to.

## Value functions

Value specs are `(func, column)` for `"first_value"` and `"last_value"`,
`(func, column, offset)` for `"lag"` and `"lead"`, and `(func, column, n)` for
`"nth_value"`, which reads the `n`-th row of the ordered partition. {py:func}`nth_value <batcher.nth_value>` is SQL
`NTH_VALUE`, and {py:func}`first_value <batcher.first_value>` is its special case `n = 1`. Use `nth_value` when the
reference point is a fixed rank, such as "each product's price relative to its category's
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

Windows may not appear where SQL also forbids them, meaning inside `group_by().agg(...)`,
in a join key, or in a sort key. Compute the window in a {py:meth}`with_columns <batcher.Dataset.with_columns>` step first,
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

`col("x").is_duplicated()` and `col("x").is_unique()` are the same idea, a
`count(1) OVER (PARTITION BY x)` compared against 1. Both are most useful inside
`filter`.

## Series recurrences

A frame answers each row from a bounded set of neighbours. Some time-series questions
can't be asked that way, because the answer depends on the whole ordered prefix through a
recurrence. Those have their own functions. All of them require `order_by`, and all of
them restart at every partition, so one device's readings never leak into another's.

`interpolate` reconstructs a continuous signal across a gap, where `forward_fill` holds
the last reading flat. Reach for the fill when the value genuinely holds between reports,
such as a configuration setting or a slowly-changing dimension, and for interpolation when
the quantity was moving the whole time, such as a temperature or a meter reading.

```python
readings = bt.from_pydict(
    {"t": [1, 2, 3, 4, 5], "temp": [10.0, None, None, 40.0, None]}
)
print(
    readings.with_columns(
        held=bt.col("temp").forward_fill().over(order_by=["t"]),
        drawn=bt.col("temp").interpolate().over(order_by=["t"]),
    ).to_pydict()
)
# {'t': [1, 2, 3, 4, 5], 'temp': [10.0, None, None, 40.0, None],
#  'held': [10.0, 10.0, 10.0, 40.0, 40.0],
#  'drawn': [10.0, 20.0, 30.0, 40.0, None]}
```

The trailing null shows the difference in kind: a fill has a value to carry, while
interpolation has nothing on the far side to draw a line to and leaves the row null.

`ewm_mean` smooths a noisy series by weighting each reading by `(1-alpha)^age`, so recent
values dominate and old ones fade instead of dropping off the cliff a fixed window has.
Spell the decay whichever way you think about it: `alpha` directly, `span` for the
N-period EMA of technical analysis, `half_life` for the lag at which a reading's weight
halves, or `com` for the centre of mass. `ewm_std` and `ewm_var` give the matching spread,
which is what makes a live volatility band or control limit.

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

`rle_id` numbers the runs of equal consecutive values, which turns a state column into
groupable segments. Group by the run id to collapse each run to a row and measure how long
it lasted.

```python
states = bt.from_pydict(
    {"t": [1, 2, 3, 4, 5, 6], "state": ["idle", "idle", "run", "run", "run", "idle"]}
)
runs = (
    states.with_columns(run=bt.col("state").rle_id().over(order_by=["t"]))
    .group_by("run")
    .agg(state=bt.col("state").min(), started=bt.col("t").min(), rows=bt.col("t").count())
    .sort("run")
)
print(runs.to_pydict())
# {'run': [0, 1, 2], 'state': ['idle', 'run', 'idle'], 'started': [1, 3, 6],
#  'rows': [2, 3, 1]}
```

A value that comes back after a gap opens a new run rather than rejoining the earlier one,
which is what makes a run id a segmentation and not a grouping.

## How a window scales across a cluster

A window is a pipeline breaker, so how it distributes depends entirely on where its work can
be cut. Batcher finds one of two seams.

The usual seam is the key. `PARTITION BY` makes every partition independent, so the rows
hash-shuffle by the partition keys and each partition is computed whole on one worker. The
union of the workers' output is the single-node answer, which is why this shape scales with
the cluster for any window function and any frame.

A window with no `PARTITION BY` has one partition over every row, so there is no key to
shuffle on. Batcher cuts along the *order* instead. It range-partitions the rows by the single
`ORDER BY` column into buckets that are ordered relative to each other, computes the window on
each bucket in parallel, then shifts each bucket's result by what the earlier buckets
contributed. A `row_number` shifts by the rows before it, a `dense_rank` by the distinct keys
before it, a running `sum` by the running total before it. Every one of those is a single
number per bucket, so the read, the partition and the window itself all divide across the
cluster.

That shift only exists for some functions. `row_number`, `rank`, `dense_rank`, running `sum`,
`count`, `min`, `max`, `avg` and `first_value` all have one. `lag`, `lead`, `last_value`,
`ntile`, `percent_rank` and `cume_dist` do not, because each reads rows or totals that its own
bucket does not hold. A window with no `PARTITION BY`, no `ORDER BY` and an aggregate is
simpler still: every row gets the same value, so it runs as an ordinary distributed aggregate
and broadcasts the scalar back.

## Requirements and limitations

- A global window distributes only when its `ORDER BY` is a single plain column, since that is
  the column the range partitioner cuts on. An expression key such as
  `order_by=[bt.col("a") + bt.col("b")]` has no distributed path.
- An explicit frame on a global window has no distributed path. Frames on a `PARTITION BY`
  window are unaffected.
- A top-N filter over a global ranking window, such as
  `.with_columns(r=bt.row_number().over(order_by="t")).filter(bt.col("r") <= 100)`, has no
  distributed path either. The filter fuses into the window as a rank bound, and a bucket
  knows only the rank within itself.
- `row_number()` over an `ORDER BY` key with duplicate values gives tied rows an arbitrary
  order, so which of them gets which number can differ between a single-node and a
  distributed run. This is true of any window, partitioned or not. Order by a key that is
  unique, or add a tiebreaker column, when the exact numbers matter.
- When a shape has no distributed path, `collect(distributed=True)` raises rather than quietly
  running the whole relation on one node. Pass `distributed=False` to ask for that explicitly.
- Out of core, a `PARTITION BY` window spills by grace-partitioning on its partition keys, and
  a global window streams bucket by bucket using the same offsets described above, so peak
  memory is one bucket rather than the whole relation.

## See also

- {doc}`Aggregations </user-guide/analyze/aggregations>`: collapse groups into summary rows.
- {doc}`Joins </user-guide/analyze/joins>`: combine windowed output with other datasets.
- {doc}`Expressions API </api/relational/expressions>`: the reference for every window, ranking
  and rolling method.
- {doc}`/cookbook/expressions/scalar/window_functions`: windows as a runnable script.
