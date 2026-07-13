# Anomaly detection

One host in the fleet started answering slowly. Find it in the metrics table, without
being told which host or when.

The textbook answer is "flag anything more than three standard deviations from the mean".
The textbook answer does not fire on this data, and the reason it does not fire is the
interesting part.

## The data

Two hosts, six minutes of latency each. Host `a` sits around 100ms and spikes to 410ms at
minute 4. Host `b` sits around 50ms and never misbehaves.

```python
import batcher as bt
from batcher import col

metrics = bt.from_pydict(
    {
        "host": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
        "minute": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        "latency": [100.0, 105.0, 98.0, 102.0, 410.0, 101.0, 50.0, 52.0, 49.0, 51.0, 53.0, 48.0],
    }
)
print(metrics.count())
# 12
```

## The trap, part one: one threshold for the whole fleet

:::{warning}
Pooling two hosts with different baselines manufactures a variance that belongs to neither
of them. One fleet-wide threshold is then far too loose for the quiet host and barely tight
enough for the busy one.
:::

```python
fleet = metrics.group_by().agg(mu=col("latency").mean(), sd=col("latency").std())
print(fleet.to_pydict())
# {'mu': [101.58333333333333], 'sd': [100.37335605657812]}
```

A fleet-wide mean of 102ms and a standard deviation of 100ms, because host `a` and host `b`
run at completely different baselines. Three sigma above the mean is 403ms. The 410ms spike
scrapes over that line by seven milliseconds, so a 400ms spike would sail straight through.
Meanwhile host `b`, whose normal is 50ms, would have to reach 403ms (eight times its own
baseline) before anyone heard about it.

Compute the baseline per host. That is a `group_by` and a join back:

```python
baseline = metrics.group_by("host").agg(mu=col("latency").mean(), sd=col("latency").std())
print(baseline.sort("host").to_pydict())
# {'host': ['a', 'b'], 'mu': [152.66666666666666, 50.5],
#  'sd': [126.08832882811426, 1.8708286933869707]}
```

## The trap, part two: the outlier eats its own evidence

Look at host `a`: mean 153ms, standard deviation 126ms. Both numbers are nonsense. `a`'s
normal latency is 98–105ms; it has a *mean* of 153 and a *sigma* of 126 only because the
410ms spike is inside the sample it is being compared against.

```python
scored = (
    metrics.join(baseline, on="host")
    .with_columns(z=(col("latency") - col("mu")) / col("sd"))
    .sort("host", "minute")
)
print(scored.select("host", "minute", "z").to_pydict()["z"])
# [-0.417696603295161, -0.3780418624760002, -0.4335584996228254, -0.4018347069674967,
#  2.0408973274928126, -0.40976565513132884, -0.2672612419124244, 0.8017837257372732,
#  -0.8017837257372732, 0.2672612419124244, 1.3363062095621219, -1.3363062095621219]
```

A 4x latency spike scores **z = 2.04**. Filter at the conventional three sigma and you get
nothing:

```python
print(scored.filter(col("z").abs() > 3.0).count())
# 0
```

:::{important}
Zero alerts. The outage is right there in the data and the detector is silent. This is
called masking, and it is not an edge case. It is what happens *every time*, because mean
and standard deviation are both computed from the point you are trying to detect. One bad
point in six can never exceed three sigma of a six-point sample: it has inflated sigma too
much.
:::

## Median and MAD

:::{tip}
Swap the mean for the median and the standard deviation for the median absolute deviation.
The median of six points does not move when one of them goes to 410; neither does the MAD.
The estimator stops defending the outlier.
:::

Two passes: the per-host median, then the median of the absolute deviations from it. The
`0.6745` factor rescales MAD so that a modified z-score is comparable to an ordinary one on
normal data (it is the 75th percentile of the standard normal).

The SQL tab runs the same three CTEs and goes one step further, applying the 3.5 threshold
in the outer `WHERE`.

::::{tab-set}
:::{tab-item} DataFrame
```python
median = metrics.group_by("host").agg(med=col("latency").median())
deviations = metrics.join(median, on="host").with_columns(
    abs_dev=(col("latency") - col("med")).abs()
)
mad = deviations.group_by("host").agg(mad=col("abs_dev").median())

robust = (
    deviations.join(mad, on="host")
    .with_columns(score=(0.6745 * (col("latency") - col("med")) / col("mad")).round(2))
    .sort("host", "minute")
)
print(robust.select("host", "minute", "latency", "score").to_pydict())
# {'host': ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'],
#  'minute': [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
#  'latency': [100.0, 105.0, 98.0, 102.0, 410.0, 101.0, 50.0, 52.0, 49.0, 51.0, 53.0, 48.0],
#  'score': [-0.4, 0.94, -0.94, 0.13, 83.23, -0.13, -0.22, 0.67, -0.67, 0.22, 1.12, -1.12]}
```
:::

:::{tab-item} SQL
```python
sql_scored = bt.sql(
    """
    WITH baseline AS (
      SELECT host, MEDIAN(latency) AS med FROM metrics GROUP BY host
    ),
    deviations AS (
      SELECT m.host, m.minute, m.latency, b.med, ABS(m.latency - b.med) AS abs_dev
      FROM metrics m INNER JOIN baseline b ON m.host = b.host
    ),
    scale AS (
      SELECT host, MEDIAN(abs_dev) AS mad FROM deviations GROUP BY host
    )
    SELECT d.host, d.minute, d.latency
    FROM deviations d INNER JOIN scale s ON d.host = s.host
    WHERE ABS(0.6745 * (d.latency - d.med) / s.mad) > 3.5
    ORDER BY d.host, d.minute
    """,
    metrics=metrics,
)
print(sql_scored.to_pydict())
# {'host': ['a'], 'minute': [4], 'latency': [410.0]}
```
:::
::::

83.2, for the same point that scored 2.04 under mean-and-sigma. Every other point is under 1.2.
Now the threshold does not need tuning, because the signal is two orders of magnitude clear of
the noise:

```python
print(robust.filter(col("score").abs() > 3.5).select("host", "minute", "latency").to_pydict())
# {'host': ['a'], 'minute': [4], 'latency': [410.0]}
```

3.5 is the usual cutoff for a modified z-score. One alert, the right one.

| | Mean and standard deviation | Median and MAD |
| --- | --- | --- |
| Score for the 410ms spike | 2.04 | 83.23 |
| Conventional threshold | 3.0 | 3.5 |
| Alerts fired | 0 | 1 |

:::{dropdown} Two caveats before you ship this
MAD can be zero. If more than half a host's readings are identical (a counter pinned at 0, a
rounded gauge) then the MAD is 0 and every score is a division by zero. Guard it with
`bt.when(col("mad") > 0).then(...).otherwise(bt.lit(0.0))`, or fall back to an interquartile
range: `col("latency").quantile(0.75) - col("latency").quantile(0.25)`.

A whole-history baseline is also not a baseline. Comparing today's latency to the median of
the last two years hides slow drift and screams at every deploy. What you usually want is a
trailing window: `col("latency").rolling_mean(60, partition_by=["host"], order_by=["minute"])`
gives a moving reference, and the deviation from *that* is what you threshold. Note that
`rolling_*` counts rows, not minutes, so the series has to be dense first (see
[time series rollups](time-series-rollups.md)). Otherwise a gap in the metrics quietly
stretches your one-hour window across a day.
:::

## See also

:::{seealso}
- [Time series rollups](time-series-rollups.md): densify the series before you window over it.
- [A/B testing](ab-testing.md): the other page here where a pooled statistic hides the
  thing you were trying to measure.
- [Aggregations](../../user-guide/aggregations.md): `median`, `quantile`, and the sketch-backed
  `approx_median` for when the groups are too big to hold.
- [Joins](../../user-guide/joins.md): the baseline-broadcast join used three times here.
- [Aggregation internals](../../deep-dives/aggregation-internals.md): why `median` is the
  expensive aggregate and how the sketch-backed version avoids it.
- [Expressions API](../../api/expressions.md): `mean`, `std`, `median`, `abs`, `round`.
:::
