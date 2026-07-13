# Time series rollups

Daily revenue, with a seven-day moving average. It is the first chart anyone builds and
it has a bug in it, because `GROUP BY day` cannot emit a day it never saw.

## The data

Six transactions across five days. Nothing happened on June 3rd: no orders, no rows, nothing
in the table to tell you so.

```python
import datetime as dt

import batcher as bt
from batcher import col

events = bt.from_pydict(
    {
        "ts": [
            dt.datetime(2024, 6, 1, 3, 15),
            dt.datetime(2024, 6, 1, 22, 4),
            dt.datetime(2024, 6, 2, 1, 0),
            dt.datetime(2024, 6, 4, 9, 30),
            dt.datetime(2024, 6, 4, 10, 0),
            dt.datetime(2024, 6, 5, 0, 5),
        ],
        "region": ["us", "eu", "us", "us", "eu", "us"],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)
print(events.count())
# 6
```

## The rollup

`dt.truncate` floors a timestamp to a unit, and `group_by` takes the derived key
directly, with no separate `with_columns` step needed. `DATE_TRUNC` in the SQL tab lowers
to the same expression.

::::{tab-set}
:::{tab-item} DataFrame
```python
daily = (
    events.group_by(day=col("ts").dt.truncate("day"))
    .agg(revenue=col("amount").sum(), orders=bt.count())
    .sort("day")
)
print(daily.to_pydict())
# {'day': [datetime.datetime(2024, 6, 1, 0, 0), datetime.datetime(2024, 6, 2, 0, 0),
#  datetime.datetime(2024, 6, 4, 0, 0), datetime.datetime(2024, 6, 5, 0, 0)],
#  'revenue': [30.0, 30.0, 90.0, 60.0], 'orders': [2, 1, 2, 1]}
```
:::

:::{tab-item} SQL
```python
sql_daily = bt.sql(
    """
    SELECT DATE_TRUNC('day', ts) AS day,
           SUM(amount) AS revenue,
           COUNT(*) AS orders
    FROM events
    GROUP BY DATE_TRUNC('day', ts)
    ORDER BY day
    """,
    events=events,
)
print(sql_daily.to_pydict()["revenue"])
# [30.0, 30.0, 90.0, 60.0]
```
:::
::::

## The trap

:::{warning}
Four rows for five days. June 3rd is not zero. It is *absent*, and those are not the same
thing. Both tabs above have the same hole in them: SQL does not save you from this one, the
spine does.
:::

Plot this and the line runs straight from June 2nd to June 4th, quietly interpolating
through the outage. Feed it to `rolling_mean(7)` and it is worse: the window counts seven
*rows*, and if a fortnight of rows is missing, "the 7-day average" silently becomes the
average of the last seven days that happened to have data. That is not a moving average,
it is a moving average of a series you do not have.

The fix is not clever. Build the days you expect, and left-join the data onto them.

## The spine

`bt.date_range` generates the calendar. Truncate it the same way you truncated the events
so the join keys have the same type, then left-join and fill.

```python
spine = bt.date_range("2024-06-01", "2024-06-05").select(day=col("date").dt.truncate("day"))
print(spine.to_pydict())
# {'day': [datetime.datetime(2024, 6, 1, 0, 0), datetime.datetime(2024, 6, 2, 0, 0),
#  datetime.datetime(2024, 6, 3, 0, 0), datetime.datetime(2024, 6, 4, 0, 0),
#  datetime.datetime(2024, 6, 5, 0, 0)]}
```

```python
dense = (
    spine.join(daily, on="day", how="left")
    .with_columns(revenue=col("revenue").fill_null(0.0), orders=col("orders").fill_null(0))
    .sort("day")
)
print(dense.to_pydict())
# {'day': [datetime.datetime(2024, 6, 1, 0, 0), datetime.datetime(2024, 6, 2, 0, 0),
#  datetime.datetime(2024, 6, 3, 0, 0), datetime.datetime(2024, 6, 4, 0, 0),
#  datetime.datetime(2024, 6, 5, 0, 0)],
#  'revenue': [30.0, 30.0, 0.0, 90.0, 60.0], 'orders': [2, 1, 0, 2, 1]}
```

Five rows, one per day, and June 3rd says zero out loud.

:::{important}
Be deliberate about the fill. Zero is right for a count or a sum, because "no orders"
really is zero revenue. It is wrong for an average, a price, or a gauge: the temperature
on a day your sensor was offline was not 0°C. For those, leave the null, or carry the last
known value forward with `col("x").forward_fill(order_by=["day"])`.
:::

## Now the moving average means something

```python
smoothed = dense.with_columns(ma3=col("revenue").rolling_mean(3, order_by=["day"]).round(2))
print(smoothed.select("day", "revenue", "ma3").to_pydict()["ma3"])
# [30.0, 30.0, 20.0, 40.0, 50.0]
```

Three rows *is* three days now, because the spine guarantees it. The leading rows average
a partial frame, which is what SQL does; pass `min_periods=3` if you would rather they be
null than half-formed.

The whole argument of the page, in one table:

| Day | `GROUP BY day` | After the spine | `ma3` |
| --- | --- | --- | --- |
| 2024-06-01 | 30.0 | 30.0 | 30.0 |
| 2024-06-02 | 30.0 | 30.0 | 30.0 |
| 2024-06-03 | *no row at all* | 0.0 | 20.0 |
| 2024-06-04 | 90.0 | 90.0 | 40.0 |
| 2024-06-05 | 60.0 | 60.0 | 50.0 |

## Rolling up by a second key

Per-region, the spine has to be the cross product of days and regions. Otherwise a region that
went quiet loses its zeros again, one region at a time.

```python
regions = bt.from_pydict({"region": ["us", "eu"]})
grid = spine.cross_join(regions)

by_region = (
    events.group_by("region", day=col("ts").dt.truncate("day"))
    .agg(revenue=col("amount").sum())
)
dense_region = (
    grid.join(by_region, on=["day", "region"], how="left")
    .with_columns(revenue=col("revenue").fill_null(0.0))
    .sort("region", "day")
)
print(dense_region.filter(col("region") == "eu").to_pydict()["revenue"])
# [20.0, 0.0, 0.0, 50.0, 0.0]
```

`eu` traded on two of the five days. Without the grid you would have got two rows and a
chart that implies a flat line.

:::{dropdown} Two more things that will bite you: time zones and late data
Time zones first. `dt.truncate("day")` floors in whatever zone the timestamps are stored in,
which for most warehouses is UTC. A "day" of revenue for a US business truncated in UTC starts
at 5pm the previous afternoon. Convert before you truncate:
`col("ts").dt.convert_timezone("UTC", "America/New_York").dt.truncate("day")`. The function is
DST-aware, so the 23-hour and 25-hour days come out right.

Then late data. A rollup run at midnight is a rollup of the events that had arrived by
midnight. If your pipeline backfills, yesterday's number changes after you published it. Either
recompute a trailing window of days on every run, or hold the bucket open with a watermark (see
[streaming](../../user-guide/streaming.md)). Decide, rather than discovering it when finance
asks why the number moved.
:::

## See also

:::{seealso}
- [Anomaly detection](anomaly-detection.md): what to do once the series is dense.
- [Sessionization](sessionization.md): the opposite problem, where the buckets have to be
  derived from the gaps rather than fixed by the calendar.
- [Window functions](../../user-guide/window-functions.md): `rolling_*`, frames, and
  `forward_fill`.
- [Joins](../../user-guide/joins.md): the left join and the cross join used here.
- [Aggregations](../../user-guide/aggregations.md): what `group_by(...).agg(...)` supports.
- [Aggregation internals](../../deep-dives/aggregation-internals.md): how the daily rollup
  merges across partitions.
- [Dataset API](../../api/dataset.md): `date_range`, `join`, `cross_join`.
:::
