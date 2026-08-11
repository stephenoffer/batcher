# Time series

This page covers the operations a time-series pipeline is built from: bucketing readings
into regular intervals, filling the intervals that have no readings, smoothing a noisy
signal, aligning two feeds that don't share a clock, and splitting a series where its
behaviour changed.

Every one of them is an ordinary relational operation over an ordered column, so they
compose with the rest of the API and run the same way on one core or a cluster. Nothing
here needs a separate time-series mode.

## Setup

One irregular sensor feed: two sensors, readings a minute apart at first and then a long
gap.

```python
import batcher as bt
import datetime as dt

base = dt.datetime(2024, 5, 1, 9, 0)
readings = bt.from_pydict(
    {
        "sensor": ["a", "a", "a", "a", "b", "b"],
        "at": [
            base,
            base + dt.timedelta(minutes=1),
            base + dt.timedelta(minutes=2),
            base + dt.timedelta(minutes=32),
            base,
            base + dt.timedelta(minutes=45),
        ],
        "celsius": [20.0, 21.0, 22.0, 26.0, 5.0, 9.0],
    }
)
```

## Downsample to regular buckets

{py:func}`bt.window(col, duration) <batcher.window>` snaps each timestamp to the start of
the interval containing it, so grouping by it is a downsample. The duration is fixed-length
(`"30m"`, `"1h"`, `"7d"`); calendar units are rejected because a month has no constant
length.

```python
bucketed = (
    readings.group_by("sensor", bucket=bt.window(bt.col("at"), "30m"))
    .agg(mean=bt.col("celsius").mean(), n=bt.col("celsius").count())
    .sort("sensor", "bucket")
)
print(bucketed.to_pydict())
# {'sensor': ['a', 'a', 'b', 'b'],
#  'bucket': [datetime.datetime(2024, 5, 1, 9, 0), datetime.datetime(2024, 5, 1, 9, 30),
#             datetime.datetime(2024, 5, 1, 9, 0), datetime.datetime(2024, 5, 1, 9, 30)],
#  'mean': [21.0, 26.0, 5.0, 9.0], 'n': [3, 1, 1, 1]}
```

Carry the count alongside the average. It is the difference between a bucket that was quiet
and a bucket where the sensor was down, and the average alone cannot tell you which.

For overlapping windows, pass `slide`: `bt.window(col("at"), "1h", "15m")` returns the
*list* of hour-wide windows a reading belongs to, hopping every fifteen minutes. Fan it out
with {py:meth}`unnest <batcher.Dataset.unnest>` before grouping.

## Fill the buckets with no rows

A group-by only emits buckets that have rows in them, so the result above jumps straight
from 09:30 to nothing. When a downstream consumer needs an unbroken grid — a plot, a join
against another series, a model that assumes fixed spacing — build the grid and left-join
onto it.

{py:func}`bt.date_range <batcher.date_range>` is the grid, and a cross join with the
distinct keys gives one row per (key, bucket).

```python
grid = bt.date_range("2024-05-01 09:00:00", "2024-05-01 10:30:00", interval="30m")
spine = grid.rename({"date": "bucket"}).cross_join(readings.select("sensor").distinct())

filled = (
    spine.join(bucketed, on=["sensor", "bucket"], how="left")
    .sort("sensor", "bucket")
    .select("sensor", "bucket", "mean", "n")
)
print(filled.to_pydict()["mean"])
# [21.0, 26.0, None, None, 5.0, 9.0, None, None]
```

The empty buckets arrive as nulls, which is the honest representation. Decide per column
what a missing bucket means, rather than letting one rule cover both:

```python
carried = filled.with_columns(
    mean=bt.col("mean").forward_fill().over(partition_by=["sensor"], order_by=["bucket"]),
    n=bt.col("n").fill_null(0),
).sort("sensor", "bucket")
print(carried.to_pydict()["mean"], carried.to_pydict()["n"])
# [21.0, 26.0, 26.0, 26.0, 5.0, 9.0, 9.0, 9.0] [3, 1, 0, 0, 1, 1, 0, 0]
```

A missing *count* is genuinely zero. A missing *temperature* is not, so it is carried
forward instead — and the row count beside it is what tells a reader the value was carried.

## Fill gaps within a series

{py:meth}`forward_fill <batcher.plan.expr_ir.core.Expr.forward_fill>` holds the last reading
flat across a gap. {py:meth}`interpolate <batcher.plan.expr_ir.core.Expr.interpolate>` draws
a straight line across it instead. Which is right depends on the quantity, not on the data:
a configuration setting or a device state genuinely holds between reports, while a
temperature or a meter reading was moving the whole time.

```python
gappy = bt.from_pydict({"at": [1, 2, 3, 4, 5], "level": [10.0, None, None, 40.0, None]})
print(
    gappy.with_columns(
        drawn=bt.col("level").interpolate().over(order_by=["at"]),
        held=bt.col("level").forward_fill().over(order_by=["at"]),
    ).to_pydict()
)
# {'at': [1, 2, 3, 4, 5], 'level': [10.0, None, None, 40.0, None],
#  'drawn': [10.0, 20.0, 30.0, 40.0, None], 'held': [10.0, 10.0, 10.0, 40.0, 40.0]}
```

The trailing null shows the difference in kind. A fill has a value to carry; interpolation
has nothing on the far side to draw a line to, so the row stays null. Both take
`partition_by` through `.over(...)`, which keeps one sensor's readings out of another's
gaps.

## Smooth over a time window

{py:meth}`rolling_mean_by <batcher.plan.expr_ir.core.Expr.rolling_mean_by>` and its family
aggregate the rows within a *duration* of the current one, rather than a fixed number of
rows. That distinction matters as soon as the sampling rate varies: a 10-row moving average
covers ten minutes when the sensor reports once a minute and three seconds when it reports
two hundred times.

```python
smoothed = readings.with_columns(
    trailing=bt.col("celsius").rolling_mean_by("at", "5m", partition_by=["sensor"]),
    seen=bt.col("celsius").rolling_count_by("at", "5m", partition_by=["sensor"]),
).sort("sensor", "at")
print(smoothed.to_pydict()["trailing"], smoothed.to_pydict()["seen"])
# [20.0, 20.5, 21.0, 26.0, 5.0, 9.0] [1, 2, 3, 1, 1, 1]
```

The fourth reading is half an hour after the third, so its five-minute window holds only
itself — and `rolling_count_by` says so.

{py:meth}`ewm_mean <batcher.plan.expr_ir.core.Expr.ewm_mean>` smooths differently: instead
of a window with a hard edge, every past reading contributes with a weight that decays as it
ages. Spell the decay however you think about it — `alpha`, `span` for the N-period EMA of
technical analysis, `half_life`, or `com`.

```python
print(
    readings.filter(bt.col("sensor") == "a")
    .with_columns(smooth=bt.col("celsius").ewm_mean(span=3).over(order_by=["at"]))
    .sort("at")
    .to_pydict()["smooth"]
)
# [20.0, 20.666666666666668, 21.42857142857143, 23.866666666666667]
```

`ewm_std` and `ewm_var` give the matching spread over the same weights, which is what makes
a live volatility band or control limit.

`ewm_mean` decays once per *row*, which is right only when the readings are evenly spaced.
For the irregular feed above it is not: the half-hour gap costs exactly the weight one
minute would. {py:meth}`ewm_mean_by <batcher.plan.expr_ir.core.Expr.ewm_mean_by>` decays by
elapsed time instead, so the smoother says the same thing whatever the sampling rate did.

```python
print(
    readings.filter(bt.col("sensor") == "a")
    .with_columns(by_time=bt.col("celsius").ewm_mean_by("at", "5m"))
    .sort("at")
    .to_pydict()["by_time"]
)
# [20.0, 20.129449436703876, 20.371591153448676, 25.912056111772635]
```

The fourth reading is half an hour after the third, so almost all of the old signal has
decayed away and the result sits close to the new reading of 26.0 — where the per-row form
above, counting that gap as one step, still returned 23.9.

## Filter by time of day

The date dominates a timestamp's ordering, so comparing timestamps cannot express "during
market hours" or "on the night shift".
{py:meth}`is_between_time <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.is_between_time>`
filters on the clock time with the date discarded, and it handles a window that wraps past
midnight — which is the case the obvious hour comparison silently returns nothing for.

```python
clock = bt.from_pydict(
    {
        "at": [
            dt.datetime(2024, 5, 1, 1, 0),
            dt.datetime(2024, 5, 1, 9, 30),
            dt.datetime(2024, 5, 1, 20, 0),
            dt.datetime(2024, 5, 2, 23, 30),
        ]
    }
)
print(
    clock.select(
        session=bt.col("at").dt.is_between_time("09:00", "17:00"),
        overnight=bt.col("at").dt.is_between_time("22:00", "02:00"),
    ).to_pydict()
)
# {'session': [False, True, False, False], 'overnight': [True, False, False, True]}
```

`bt.col("at").dt.hour() >= 22` combined with `<= 2` returns nothing at all for that second
window, which is why the wrap is handled in one tested place rather than at each call site.

{py:meth}`time_of_day <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.time_of_day>`
gives the same clock reading as a number — microseconds since midnight — which is what you
group by to compare a load curve across days, or subtract to get "minutes into the session".

## Align two series with different clocks

Two feeds almost never share a clock, so an equi-join on the timestamp finds nothing.
{py:meth}`join_asof <batcher.Dataset.join_asof>` matches each left row to the nearest right
row instead, and `tolerance` bounds how stale that match may be.

```python
trades = bt.from_pydict({"sym": ["A", "A"], "t": [10, 40], "size": [100, 200]})
quotes = bt.from_pydict({"sym": ["A", "A"], "t": [8, 12], "price": [1.0, 1.1]})
print(trades.join_asof(quotes, on="t", by="sym", tolerance=5).sort("t").to_pydict())
# {'sym': ['A', 'A'], 't': [10, 40], 'size': [100, 200], 'price': [1.0, None]}
```

The second trade's nearest preceding quote is 28 units old, so the tolerance leaves it
unmatched rather than pricing it against a stale quote. Without the tolerance it would have
matched, silently. See {doc}`/user-guide/analyze/joins` for `direction` and for
`allow_exact_matches`, the guard against look-ahead bias in a backtest.

Both forms distribute. With `by`, both sides are co-partitioned on the `by` keys, so each
worker holds whole groups. Without it, both sides are range-partitioned on the `on` key
through one shared boundary list, and each partition is additionally lent the single row on
each side of its boundary that could still win a match. Either way the answer is the one node
computes; see {doc}`/architecture/deep-dives/operators/join-algorithms` for why one carried
row per direction is enough.

## Split a series where it changed

{py:meth}`rle_id <batcher.plan.expr_ir.core.Expr.rle_id>` numbers the runs of equal
consecutive values, which turns a state column into groupable segments. Group by the run id
to collapse each run to one row and measure how long it lasted.

```python
states = bt.from_pydict(
    {"at": [1, 2, 3, 4, 5, 6], "state": ["ok", "ok", "fault", "fault", "fault", "ok"]}
)
runs = (
    states.with_columns(run=bt.col("state").rle_id().over(order_by=["at"]))
    .group_by("run")
    .agg(state=bt.col("state").min(), start=bt.col("at").min(), length=bt.col("at").count())
    .sort("run")
)
print(runs.to_pydict())
# {'run': [0, 1, 2], 'state': ['ok', 'fault', 'ok'], 'start': [1, 3, 6], 'length': [2, 3, 1]}
```

A value that comes back after an interruption opens a *new* run rather than rejoining the
earlier one, which is what makes a run id a segmentation rather than a grouping.

## Requirements and limitations

- Every operation on this page that carries a value along an order — the fills,
  `interpolate`, `rle_id`, the EWM family — requires `order_by`. An unordered relation has
  no "previous row", and a morsel-parallel or distributed scan will not supply one.
- Window durations are fixed-length. `"1mo"` and `"1y"` are rejected for window widths and
  for a rolling `window_size`, because they have no constant microsecond length. Use
  {py:meth}`.dt.truncate("month") <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.truncate>`
  and group on the result for calendar buckets.
- A `"range"` frame offset, `rolling_*_by`, an ASOF `tolerance`, and
  `direction="nearest"` all need a numeric or temporal key, because each one subtracts two
  keys. A string key still orders fine for everything that only compares.
- Temporal offsets are in **microseconds** wherever a number is passed instead of a
  duration string, whatever resolution the column is stored at.

## See also

- {doc}`/user-guide/analyze/window-functions`: frames, ranking, and the full window vocabulary.
- {doc}`/user-guide/analyze/joins`: as-of joins in full, including `direction` and `tolerance`.
- {doc}`/user-guide/moving-data/streaming`: the same operations over an unbounded source,
  with watermarks and triggers.
- {doc}`/cookbook/analytics/aggregates/time-series-rollups`: these patterns as a runnable script.
