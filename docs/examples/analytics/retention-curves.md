# Retention curves

Of the people who signed up on Monday, how many came back on day 1? Day 7? Day 30? A
retention curve is a fraction, and every way of getting it wrong is a way of getting the
numerator and the denominator out of alignment. The tell is a retention rate above 100%,
which is common enough that most analysts have shipped one.

## The data

Ten activity rows, four users. `u1` signs up on May 1st and is active twice that day, which is
the detail that breaks the naive query.

```python
import datetime as dt

import batcher as bt
from batcher import col

def may(day: int) -> dt.date:
    return dt.date(2024, 5, day)


activity = bt.from_pydict(
    {
        "user": ["u1", "u1", "u1", "u2", "u2", "u3", "u4", "u4", "u1", "u2"],
        "day": [may(1), may(1), may(2), may(1), may(8), may(1), may(2), may(9), may(8), may(2)],
    }
)
print(activity.count())
# 10
```

## Cohort and day number

Same skeleton as [cohort analysis](cohort-analysis.md), in days rather than months. The
cohort is the user's first active day; `day_n` is days since then.

Dates do not subtract and window aggregates do not reduce a `Date32`, so convert to an
integer day number once, up front. `dt.epoch()` gives seconds; divide by 86,400 and cast.
The string form of the date goes along for the ride as a readable cohort label. `min` over a
string is well defined, and ISO dates sort correctly as text.

```python
SECONDS_PER_DAY = 86_400

labelled = (
    activity.with_columns(
        day_num=(col("day").dt.epoch() // SECONDS_PER_DAY).cast("int64"),
        day_str=col("day").dt.strftime("%Y-%m-%d"),
    )
    .with_columns(
        first_num=col("day_num").min().over(partition_by=["user"]),
        cohort=col("day_str").min().over(partition_by=["user"]),
    )
    .with_columns(day_n=col("day_num") - col("first_num"))
)
print(labelled.sort("user", "day").select("user", "cohort", "day_n").to_pydict())
# {'user': ['u1', 'u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u4', 'u4'],
#  'cohort': ['2024-05-01', '2024-05-01', '2024-05-01', '2024-05-01', '2024-05-01',
#             '2024-05-01', '2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02'],
#  'day_n': [0, 0, 1, 7, 0, 1, 7, 0, 0, 7]}
```

## The trap

:::{warning}
`COUNT(*)` counts events. Retention is about *people*. Those are the same number only when
every user appears exactly once per day, which is true in test fixtures and never true in
production.
:::

Count the rows in each cell:

```python
naive = (
    labelled.group_by("cohort", "day_n")
    .agg(active=bt.count())
    .sort("cohort", "day_n")
)
print(naive.to_pydict())
# {'cohort': ['2024-05-01', '2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02'],
#  'day_n': [0, 1, 7, 0, 7], 'active': [4, 2, 2, 1, 1]}
```

Four "active users" on day 0 of the May 1st cohort. There are three. `u1` logged in twice
and got counted twice, so day-0 retention comes out as 4/3 = 133%.

| Cohort, day | `bt.count()` rows | `n_unique()` people |
| --- | --- | --- |
| 2024-05-01, day 0 | 4 | 3 |
| 2024-05-01, day 1 | 2 | 2 |
| 2024-05-01, day 7 | 2 | 2 |
| 2024-05-02, day 0 | 1 | 1 |
| 2024-05-02, day 7 | 1 | 1 |

One cell differs, and it is the denominator cell. That is enough to put the whole curve
above 100%.

## Count people

`n_unique` on the user column, and the denominator taken from `day_n == 0`: the cohort's own
size, not the total number of active users, not the size of the largest cohort. The SQL tab
keys the cohort on the raw day number rather than a formatted date, because an integer is a
cheaper shuffle key than a string and nothing downstream cares what it looks like until you
render it.

::::{tab-set}
:::{tab-item} DataFrame
```python
cells = labelled.group_by("cohort", "day_n").agg(active=col("user").n_unique())
sizes = (
    labelled.filter(col("day_n") == 0)
    .group_by("cohort")
    .agg(cohort_size=col("user").n_unique())
)

curve = (
    cells.join(sizes, on="cohort")
    .with_columns(retention=(col("active") / col("cohort_size")).round(3))
    .sort("cohort", "day_n")
)
print(curve.to_pydict())
# {'cohort': ['2024-05-01', '2024-05-01', '2024-05-01', '2024-05-02', '2024-05-02'],
#  'day_n': [0, 1, 7, 0, 7], 'active': [3, 2, 2, 1, 1], 'cohort_size': [3, 3, 3, 1, 1],
#  'retention': [1.0, 0.667, 0.667, 1.0, 1.0]}
```
:::

:::{tab-item} SQL
```python
sql_curve = bt.sql(
    """
    WITH labelled AS (
      SELECT user,
             CAST(EXTRACT(EPOCH FROM day) / 86400 AS BIGINT) AS day_num
      FROM activity
    ),
    cohorted AS (
      SELECT user, day_num,
             MIN(day_num) OVER (PARTITION BY user) AS first_num
      FROM labelled
    ),
    cells AS (
      SELECT first_num AS cohort, day_num - first_num AS day_n,
             COUNT(DISTINCT user) AS active
      FROM cohorted
      GROUP BY first_num, day_num - first_num
    ),
    sizes AS (
      SELECT first_num AS cohort, COUNT(DISTINCT user) AS cohort_size
      FROM cohorted
      WHERE day_num = first_num
      GROUP BY first_num
    )
    SELECT c.cohort, c.day_n, c.active,
           CAST(c.active AS DOUBLE) / s.cohort_size AS retention
    FROM cells c INNER JOIN sizes s ON c.cohort = s.cohort
    ORDER BY c.cohort, c.day_n
    """,
    activity=activity,
)
print(sql_curve.to_pydict()["retention"])
# [1.0, 0.6666666666666666, 0.6666666666666666, 1.0, 1.0]
```
:::
::::

:::{tip}
Day 0 is 100% by construction, which is the sanity check. If it is not exactly 1.0, your
cohort definition and your denominator disagree and everything downstream is wrong.
:::

## Read the curve as a curve

`pivot` puts the days across the top:

```python
grid = curve.pivot(index=["cohort"], on="day_n", values="retention", aggregate="max").sort("cohort")
print(grid.to_pydict())
# {'cohort': ['2024-05-01', '2024-05-02'], '0': [1.0, 1.0], '1': [0.667, None],
#  '7': [0.667, 1.0]}
```

Two things to notice, and neither is a bug.

:::{important}
The nulls are not zeros. The May 2nd cohort has no day-1 number because none of its users
were active on day 1, and here that genuinely means zero. But if the cohort were three days
old and you asked for day 7, the null would mean *not yet observable*, and filling it with 0
would drag your average retention curve downwards for no reason at all. Right-censoring is
the technical name, and forgetting about it is the most common way retention charts lie.
:::

The curve is also not monotonic. The May 1st cohort holds 66.7% at both day 1 and day 7,
because users come back after a gap. Day-N retention measures activity *on that exact day*, so
it bounces. If you want a curve that only falls, you want rolling retention instead (active on
day N *or later*), which is what "N-day retention" means at some companies and not at others.
Say which one you are reporting. The two can differ by a factor of two on the same data.

:::{dropdown} Scaling notes: swapping the exact distinct count for a sketch
`n_unique` is an exact distinct count, which means it holds every distinct user id per cell
in memory. On a cohort table with hundreds of millions of users that is the step that hurts.
`approx_n_unique` swaps it for a HyperLogLog sketch: bounded memory per group, ~2% error,
and mergeable, so the answer is identical single-node and distributed. For a retention
*curve*, 2% is well inside the noise you already have.

```python
approx = (
    labelled.group_by("cohort", "day_n")
    .agg(active=col("user").approx_n_unique())
    .sort("cohort", "day_n")
)
print(approx.to_pydict()["active"])
# [3, 2, 2, 1, 1]
```
:::

## See also

:::{seealso}
- [Cohort analysis](cohort-analysis.md): the same skeleton, measured in months and revenue.
- [A/B testing](ab-testing.md): the other page where the denominator has to come from the
  assignment rather than from the behaviour.
- [Aggregations](../../user-guide/aggregations.md): `n_unique` and its sketch-backed twin.
- [Window functions](../../user-guide/window-functions.md): `min().over(...)` and the rest.
- [Pivoting](../../user-guide/pivoting.md): laying the days out across the top.
- [Aggregation internals](../../deep-dives/aggregation-internals.md): why the HyperLogLog
  sketch merges across partitions and the exact count does not.
- [Dataset API](../../api/dataset.md): `group_by`, `join`, `pivot`.
:::
