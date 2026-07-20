# Cohort analysis

Do customers acquired in January spend more in their third month than customers
acquired in March? That is a cohort question, and the whole thing turns on one
column: which cohort a row belongs to.

Rows do not have cohorts. *Users* have cohorts. A row's cohort is the month of its
user's **first** order, which is not a property of the row you are looking at.

## The data

```python
import datetime as dt

import batcher as bt
from batcher import col

orders = bt.from_pydict(
    {
        "user": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
        "order_date": [
            dt.date(2024, 1, 5),
            dt.date(2024, 2, 9),
            dt.date(2024, 4, 2),
            dt.date(2024, 1, 20),
            dt.date(2024, 3, 3),
            dt.date(2024, 2, 14),
            dt.date(2024, 2, 27),
            dt.date(2024, 3, 8),
        ],
        "amount": [50.0, 30.0, 20.0, 80.0, 40.0, 25.0, 15.0, 60.0],
    }
)
print(orders.count())
# 8
```

## The trap

:::{warning}
Grouping by the order's own month puts one user in three different cohorts. `u1` bought
in January, February and April, so it lands in all three, and every retention number
computed on top of that inflates.
:::

Group by the order's own month and you get a monthly sales report, not a cohort
report:

```python
naive = (
    orders.group_by(month=col("order_date").dt.strftime("%Y-%m"))
    .agg(users=col("user").n_unique(), revenue=col("amount").sum())
    .sort("month")
)
print(naive.to_pydict())
# {'month': ['2024-01', '2024-02', '2024-03', '2024-04'], 'users': [2, 2, 2, 1],
#  'revenue': [130.0, 70.0, 100.0, 20.0]}
```

Four cohorts, seven user-slots, four actual users. Sum the `users` column and you
get 7, which is more people than exist. Every retention number computed on top of
this is wrong in the same direction: it inflates.

Follow `u1`'s three orders through both labellings and the difference is the whole page:

| `u1`'s order | Naive: the order's own month | Correct: the user's first month |
| --- | --- | --- |
| 2024-01-05 | 2024-01 | 2024-01 |
| 2024-02-09 | 2024-02 | 2024-01 |
| 2024-04-02 | 2024-04 | 2024-01 |

## Label the user, not the row

:::{tip}
The cohort is `min(order_date)` **per user**, which is a window aggregate rather than a
group aggregate. You want the label attached back to every row of that user instead of
collapsed, and `.over(partition_by=["user"])` does exactly that.
:::

`month_idx` is the month as a single integer (`year * 12 + month`), so subtracting
two of them gives the number of months between, with no calendar arithmetic and no
December-to-January wraparound bug.

```python
labelled = (
    orders.with_columns(
        month=col("order_date").dt.strftime("%Y-%m"),
        month_idx=col("order_date").dt.year() * 12 + col("order_date").dt.month(),
    )
    .with_columns(
        cohort=col("month").min().over(partition_by=["user"]),
        cohort_idx=col("month_idx").min().over(partition_by=["user"]),
    )
    .with_columns(period=col("month_idx") - col("cohort_idx"))
)
print(labelled.sort("user", "order_date").select("user", "month", "cohort", "period").to_pydict())
# {'user': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4'],
#  'month': ['2024-01', '2024-02', '2024-04', '2024-01', '2024-03', '2024-02', '2024-02', '2024-03'],
#  'cohort': ['2024-01', '2024-01', '2024-01', '2024-01', '2024-01', '2024-02', '2024-02', '2024-03'],
#  'period': [0, 1, 3, 0, 2, 0, 0, 0]}
```

Now each user carries one cohort for its whole life, and `period` is months since
acquisition. `u1`'s April order sits at period 3 of the January cohort, where it
belongs.

## The cohort table

Batcher lowers SQL and the DataFrame API to one logical plan, so the two tabs below are
the same query, not a second implementation of it. The SQL cohort label is `YYYYMM` as an
integer, which sorts the same way the string does and needs no format function.

::::{tab-set}
:::{tab-item} DataFrame
```python
cells = (
    labelled.group_by("cohort", "period")
    .agg(users=col("user").n_unique(), revenue=col("amount").sum())
    .sort("cohort", "period")
)
print(cells.to_pydict())
# {'cohort': ['2024-01', '2024-01', '2024-01', '2024-01', '2024-02', '2024-03'],
#  'period': [0, 1, 2, 3, 0, 0], 'users': [2, 1, 1, 1, 1, 1],
#  'revenue': [130.0, 30.0, 40.0, 20.0, 40.0, 60.0]}
```
:::

:::{tab-item} SQL
```python
sql_cells = bt.sql(
    """
    WITH labelled AS (
      SELECT user, amount,
             EXTRACT(YEAR FROM order_date) * 100 + EXTRACT(MONTH FROM order_date) AS month,
             EXTRACT(YEAR FROM order_date) * 12 + EXTRACT(MONTH FROM order_date) AS month_idx
      FROM orders
    ),
    cohorted AS (
      SELECT user, amount, month_idx,
             MIN(month) OVER (PARTITION BY user) AS cohort,
             MIN(month_idx) OVER (PARTITION BY user) AS cohort_idx
      FROM labelled
    )
    SELECT cohort, month_idx - cohort_idx AS period,
           COUNT(DISTINCT user) AS users, SUM(amount) AS revenue
    FROM cohorted
    GROUP BY cohort, month_idx - cohort_idx
    ORDER BY cohort, period
    """,
    orders=orders,
)
print(sql_cells.to_pydict())
# {'cohort': [202401, 202401, 202401, 202401, 202402, 202403],
#  'period': [0, 1, 2, 3, 0, 0], 'users': [2, 1, 1, 1, 1, 1],
#  'revenue': [130.0, 30.0, 40.0, 20.0, 40.0, 60.0]}
```
:::
::::

The January cohort has two users and keeps one of them alive through month 3. The
February and March cohorts have not had time to show anything yet, a fact the flat list hides
and the triangle makes obvious.

## The triangle

`pivot` spreads `period` across the columns, which is how anyone actually reads a
cohort report:

```python
triangle = cells.pivot(index=["cohort"], on="period", values="users", aggregate="sum").sort("cohort")
print(triangle.to_pydict())
# {'cohort': ['2024-01', '2024-02', '2024-03'], '0': [2, 1, 1],
#  '1': [1, None, None], '2': [1, None, None], '3': [1, None, None]}
```

:::{important}
Do not fill the nulls in the lower-right with zeros. A zero means "nobody came back". A
null means "not known yet", and averaging a column that mixes the two is how a
retention chart starts trending down for no reason.
:::

The nulls are the shape of the thing: the March cohort has no month-3 number because
month 3 has not happened.

`pivot` runs an eager pre-pass to discover the distinct values of `period`. Pass
`columns=[0, 1, 2, 3]` to fix them yourself and skip that pass, which is worth doing when the
period range is known and the input is large.

:::{dropdown} Scaling notes: the shuffle that decides whether this fits in memory
The `min().over(partition_by=["user"])` is a hash shuffle on `user`. It is the expensive step,
and the one that decides whether this query fits in memory. Two things keep it bounded:

- Project first. `orders` here has three columns, while a real event table has forty. Select
  the three you need *before* the window, so the shuffle moves bytes you will use.
- If `user` is skewed (one bot account with a million orders), the window partition for
  that key is what spills. Batcher's aggregates and windows spill to disk rather than
  dying, but a skewed key is still slow. `col("user").n_unique()` on the raw table tells
  you before you find out the hard way.
:::

## See also

:::{seealso}
- [Retention curves](retention-curves.md): the same cohort skeleton, measured in days
  and normalized to a rate.
- [Funnel analysis](funnel-analysis.md): the other one-row-per-user collapse, and the
  self-join it replaces.
- [Window functions](../../user-guide/window-functions.md): `over` in full.
- [Aggregations](../../user-guide/aggregations.md): `n_unique` and the approximate
  variant for large inputs.
- [Pivoting](../../user-guide/pivoting.md): `pivot`, `unpivot`, and fixing the column set.
- [Window internals](../../deep-dives/window-internals.md): what the partition-by shuffle
  actually costs, and when it spills.
- [Expressions API](../../api/expressions.md): `dt.strftime`, `dt.year`, `min().over(...)`.
:::
