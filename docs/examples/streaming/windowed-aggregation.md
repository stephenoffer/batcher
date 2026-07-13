# Windowed aggregation

A `GROUP BY` over a bounded table finishes. Over a stream it never does: every group
stays live forever, because another row for it might arrive tomorrow.

:::{warning}
Run a plain `group_by("user").agg(...)` against Kafka in `complete` mode and the state grows
for as long as the job runs. Eventually it is the job's memory that ends the query.
:::

Event-time windows fix this by making a group *finishable*. A window has an end. Once
the watermark says no more rows for that window will arrive, the window is emitted and
its state is freed. Windows plus a watermark is the only combination that gives you
bounded state and `append` output.

## The window is the same in batch and streaming

`bt.window(time_col, duration)` assigns a row to a window; group by it like any other
key. Nothing about the aggregation changes between the two tabs below: the source changes,
and a watermark is declared.

::::{tab-set}
:::{tab-item} A bounded table

On a bounded table it is a normal aggregation:

```python
import datetime as dt

import batcher as bt
from batcher import col

base = dt.datetime(2024, 1, 1)
minute = dt.timedelta(minutes=1)

events = bt.from_pydict({
    "ts": [base, base + 30 * minute, base + 90 * minute, base + 100 * minute],
    "user": ["a", "b", "a", "c"],
    "amount": [3, 5, 7, 11],
})

hourly = events.group_by(w=bt.window(col("ts"), "1h")).agg(total=col("amount").sum())
print(hourly.to_pydict())
# {'w': [datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 1, 1, 0)], 'total': [8, 18]}
```

Two windows, two rows, done. Develop the aggregation here: it is the same operator the
streaming path runs, so if the numbers are right on a fixture they are right on the
stream.
:::

:::{tab-item} An unbounded source

Now the source never ends. Add `.with_watermark(time_col, lateness)` and the identical
`group_by(...).agg(...)` becomes a windowed streaming aggregation: state is one running
partial per open window, and a window is emitted the moment the watermark passes its
end.

The watermark is `max(event_time seen) - lateness`. Watch it work. The third micro-batch
carries an event at 02:10, which pushes the watermark to 02:00 and closes the 00:00
window:

```python
import pyarrow as pa

schema = pa.schema([("ts", pa.timestamp("us")), ("amount", pa.int64())])


def feed():
    yield pa.record_batch({"ts": [base, base + 30 * minute], "amount": [3, 5]}, schema=schema)
    yield pa.record_batch({"ts": [base + 130 * minute], "amount": [1]}, schema=schema)


stream = bt.from_batches(feed, schema, bounded=False)

windowed = (
    stream.with_watermark("ts", "10m")
    .group_by(w=bt.window(col("ts"), "1h"))
    .agg(total=col("amount").sum())
)
for batch in windowed.iter_batches():
    print(batch.to_pydict())
# {'w': [datetime.datetime(2024, 1, 1, 0, 0)], 'total': [8]}
# {'w': [datetime.datetime(2024, 1, 1, 2, 0)], 'total': [1]}
```

The 00:00 window emitted while the stream was still running. The 02:00 window was still
open when the feed ended, so it came out in the end-of-stream flush. On a real topic that
flush happens when the query stops; until then, an open window sits in state.
:::
::::

## Wiring it to a sink

`output_mode="append"` on a windowed aggregation emits a window's row once, when the
watermark closes it. That is what a downstream table wants: no restatements, no
upserts.

:::{dropdown} The full Kafka → Delta gold query

```python
# docs: skip
query = (
    bt.read.kafka("clicks", bootstrap_servers="broker-1:9092")
    .with_columns(payload=col("value").cast("string"))
    .select(
        ts=col("payload").json.extract_string("$.ts").cast("timestamp"),
        amount=col("payload").json.extract_int("$.amount"),
    )
    .with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1 hour"))
    .agg(hits=col("amount").count(), total=col("amount").sum())
    .write.delta(
        "s3://lake/gold/hourly_clicks",
        trigger=bt.Trigger.processing_time("1 minute"),
        output_mode="append",
        checkpoint="s3://lake/gold/hourly_clicks/_ckpt",
        query_name="gold-hourly-clicks",
    )
)
query.await_termination()
```
:::

The three output modes are not interchangeable, and the engine will tell you so:

| `output_mode` | What a micro-batch emits | The catch |
| --- | --- | --- |
| `"append"` | a window's row once, when the watermark closes it | needs a watermark on a windowed aggregation; ask for it on a plain `group_by("user")` and you get a `PlanError`, because the engine cannot know a group is final |
| `"complete"` | the whole result table, every micro-batch | right for a small, bounded key space (a dashboard of 50 regions), wrong for unbounded keys, because every group is retained forever |
| `"update"` | only the rows that changed | your sink must be able to upsert |

`Trigger.continuous(...)` does not run aggregations at all; it is stateless pipelines
only. Use a processing-time trigger.

## Sessions

Windows with fixed boundaries do not describe user behavior. `session_window` groups
consecutive events per key whose gap is under a timeout, and starts a new session when
the gap is exceeded. Same aggregate expressions:

```python
sessions = events.session_window(
    "ts", "45m", partition_by=["user"], total=col("amount").sum()
)
print(sessions.select("user", "session_start", "total").to_pydict()["total"])
# [3, 5, 7, 11]
```

User `a` clicked at 00:00 and again at 01:30, a 90-minute gap, so two sessions (3 and 7),
not one of 10.

## What bounds the state, and what happens when it doesn't

Memory for a windowed streaming aggregate is proportional to the number of *open*
windows, and windows close only when the watermark advances. The watermark advances only
when event time advances. So an idle partition, a clock skew, or a source that stops
producing will stall the watermark, and open windows accumulate.

Batcher caps that state and fails loudly rather than dying by OOM: the retained state is
checked against `memory.streaming_state_max_bytes` and a `ResourceError` names the column
whose watermark is not advancing. That is a real signal, not a tuning knob to raise
reflexively. If it fires, the usual cause is an event-time gap or a dead partition, not
an undersized budget.

:::{important}
There is no side output for the rows the watermark drops, and no per-batch late-row
counter in `recent_progress()`. A window that closed early is quietly short, and nothing in
the job's metrics says so. Late data is covered in
[Late data and watermarks](late-data-watermarks.md).
:::

## See also

- [Late data and watermarks](late-data-watermarks.md): what `lateness` actually costs.
- [Exactly-once sinks](exactly-once-sink.md): making the gold write survive a restart.
- [Kafka to the lake](kafka-etl.md): the bronze layer this gold one aggregates.
- [Streaming](../../user-guide/streaming.md): triggers, output modes, and the query handle.
- [Aggregations](../../user-guide/aggregations.md): the aggregate surface itself.
- [Window functions](../../user-guide/window-functions.md): the other kind of window, over a
  bounded frame.
- [Delta Lake integration](../../integrations/delta-lake.md): the sink in the query above.
- [Mergeable algebra](../../deep-dives/mergeable-algebra.md): why one running partial per open
  window is all the state there is.
- [Time-series rollups](../analytics/time-series-rollups.md): the same windows, computed as a
  batch.
