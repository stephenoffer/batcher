# Late-arriving data

A phone was in airplane mode. Its events happened on Tuesday and reached your ingest on
Thursday. Tuesday's partition was written Tuesday night, Tuesday's revenue number was
published Wednesday morning, and both are now missing seven dollars.

Every stream has this. The only question is whether your pipeline can repair itself when
it happens, or whether it needs a human.

## Rule zero: partition by event time

:::{warning}
The mistake that cannot be fixed later is partitioning by *processing* time. Do that, and
Tuesday's events physically live in Thursday's partition. "Revenue on Tuesday" is now a
full-table scan at best, and wrong at worst, and no backfill helps because the partition
key does not encode the fact you care about.
:::

Partition by event time. Then late data has a home: it belongs to a day you already
wrote, and the repair job knows exactly which day to rebuild.

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
raw = os.path.join(work, "events")

bt.from_pydict(
    {
        "event_id": [1, 2, 3],
        "event_day": ["2024-03-05", "2024-03-05", "2024-03-06"],
        "revenue": [10.0, 20.0, 5.0],
    }
).write.delta(raw, mode="overwrite")
```

The derived table everyone actually queries:

```python
daily = os.path.join(work, "daily")


def rebuild(days):
    """Recompute the daily rollup for exactly `days`, replacing those rows and no others."""
    for day in days:
        agg = (
            bt.read.delta(raw)
            .filter(bt.col("event_day") == day)
            .group_by("event_day")
            .agg(revenue=bt.col("revenue").sum())
        )
        if os.path.exists(daily):
            agg.write.delta(
                daily,
                replace_where=bt.col("event_day") == day,
                partition_by=["event_day"],
            )
        else:
            agg.write.delta(daily, mode="overwrite", partition_by=["event_day"])


rebuild(["2024-03-05", "2024-03-06"])
print(bt.read.delta(daily).sort("event_day").to_pydict())
# {'event_day': ['2024-03-05', '2024-03-06'], 'revenue': [30.0, 5.0]}
```

## Repair, not patch

Thursday's batch shows up carrying one Tuesday event and one fresh one. Land it in raw as
usual:

```python
late = bt.from_pydict(
    {"event_id": [4, 5], "event_day": ["2024-03-05", "2024-03-07"], "revenue": [7.0, 3.0]}
)
late.write.delta(raw, mode="append")
```

Now: which days did that batch touch? Ask the batch, in the engine, not by iterating rows
in Python:

```python
affected = late.distinct(subset=["event_day"]).sort("event_day").to_pydict()["event_day"]
print(affected)
# ['2024-03-05', '2024-03-07']
```

That is a small list of partition keys crossing into the control plane, which is exactly
what the control plane is for. Recompute those days from raw and replace them:

:::{note}
`replace_where=` has to resolve to partition filters delta-rs can act on from the log
alone, which means an AND of `partition_col == value`. A multi-value `is_in([...])`
is not expressible that way, so `rebuild` loops and issues one partition-scoped overwrite
per day. Each one is still a metadata operation that moves no data, so the loop costs about
what a single call would.
:::

```python
rebuild(affected)
print(bt.read.delta(daily).sort("event_day").to_pydict())
# {'event_day': ['2024-03-05', '2024-03-06', '2024-03-07'], 'revenue': [37.0, 5.0, 3.0]}
```

Tuesday went from 30 to 37. Wednesday was not touched (not recomputed, not rewritten, not
re-read). Thursday appeared. The rollup is now exactly what a from-scratch rebuild would
produce, at the cost of the days that actually changed.

:::{important}
Do not be tempted to *add* the late revenue to the existing total. Incrementally patching
an aggregate works for `sum` and `count` and falls apart the moment somebody asks for a
`median` or a distinct count, and it is not idempotent: run the patch twice and Tuesday is
44. Recompute the affected slice from the source of truth. It is more I/O and it is always
right.
:::

See {doc}`partition backfill <partition-backfill>` for the mechanics of the replace.

## The streaming side: watermarks

A streaming aggregate cannot wait forever for a straggler. It has to emit the 10:00 window
at some point, and it has to forget the state that window used, or memory grows without
bound and the job dies on a Sunday.

A watermark is where you write that decision down. `with_watermark(time_col, lateness)`
says: once I have seen event time T, I will accept rows back to `T - lateness` and drop
anything older. The query is the same either way. What changes is what the watermark is
actually doing.

::::{tab-set}

:::{tab-item} Bounded source

```python
import datetime

t0 = datetime.datetime(2024, 3, 5, 10, 0)
minute = datetime.timedelta(minutes=1)

clicks = bt.from_pydict(
    {"ts": [t0, t0 + minute, t0 + 6 * minute], "revenue": [1.0, 2.0, 4.0]}
)
windowed = (
    clicks.with_watermark("ts", "10m")
    .group_by(w=bt.window(bt.col("ts"), "5m"))
    .agg(revenue=bt.col("revenue").sum())
)
print(windowed.sort("w").to_pydict())
# {'w': [datetime.datetime(2024, 3, 5, 10, 0), datetime.datetime(2024, 3, 5, 10, 5)],
#  'revenue': [3.0, 4.0]}
```

Two five-minute windows, bucketed by event time and not by when the row showed up. On a
bounded source such as this one, the watermark changes nothing: everything is already here.
:::

:::{tab-item} Kafka stream

```python
# docs: skip
query = (
    bt.read.kafka("clicks", bootstrap_servers="broker:9092")
    .with_watermark("ts", "10m")
    .group_by(w=bt.window(bt.col("ts"), "5m"))
    .agg(revenue=bt.col("revenue").sum())
    .write("s3://lake/clicks_5m", trigger=bt.Trigger.processing_time("1 minute"))
)
```

Here the watermark is doing the actual work: it decides when the 10:00 window is final and
its state can be evicted, and it decides which stragglers are dropped rather than counted.
:::

::::

Pick the lateness from the delivery-delay distribution you actually observe, not from a
number that sounds tidy. Ten minutes of allowed lateness means ten minutes of state per
open window, and an hour means an hour of it. That is the real cost, and it is why "set it
to a day" is not free.

## You cannot have both

:::{important}
Bounded state and unbounded lateness are incompatible. A watermark buys you the first by
giving up the second, and the straggler that arrives after it passes is *gone from the
streaming aggregate*. It is not counted late. It is not counted at all. There is no configuration
that avoids this trade.
:::

The workable shape is to accept it on purpose: the streaming aggregate is the fast,
slightly-lossy answer at the tail, and a batch job reconciles the affected days from raw on
a schedule (that is the `rebuild` above, run nightly over the last N days). Raw keeps
everything, so the batch pass is exact and idempotent, and the numbers converge.

Say out loud which of your tables is which. A dashboard reading the streaming rollup and a
finance report reading the reconciled one will disagree for a few hours, and that is fine
as long as everyone knows why.

| | Streaming aggregate | Nightly rebuild from raw |
|---|---|---|
| Latency | seconds | a day, or whatever the schedule says |
| State | bounded by the lateness you allow | none; it reads the source of truth |
| A straggler past the watermark | dropped | counted |
| Rerunning it | changes the answer as new data lands | converges on the same answer |
| What it is for | the dashboard | the number someone will be held to |

## See also

- {doc}`Partition backfill <partition-backfill>`: the atomic replace that `rebuild` leans on.
- {doc}`Deduplication <deduplication>`: late rows are often duplicate rows too.
- {doc}`CDC pipeline <cdc-pipeline>`: out-of-order changes, sequenced rather than windowed.
- {doc}`Streaming <../../user-guide/streaming>`: triggers, output modes, and the streaming
  query handle.
- {doc}`Lakehouse tables <../../user-guide/lakehouse>`: the raw table the rebuild reads from.
- {doc}`Kafka <../../integrations/kafka>`: the stream the stragglers arrive on.
- {doc}`Dataset API <../../api/dataset>`: `with_watermark`, `window`, `group_by`.

