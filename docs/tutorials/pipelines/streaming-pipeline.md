# A streaming pipeline

Build a continuous pipeline: read an unbounded stream, dedupe it, window it by event time,
and write each micro-batch out. The point of this tutorial is what you *don't* have to
learn. Batch is the bounded special case of streaming, so the operators are the ones you
already know, and only the source and the trigger change.

Everything here runs as written, with a generator standing in for Kafka.

:::{note}
**What you'll build.** An unbounded source, a watermarked dedup, an event-time window, a
triggered write, a checkpointed file sink, and a custom per-batch hook. You need `pip install
batcher-engine` and nothing else: a generator stands in for Kafka, and no broker is required.
:::

| You need | For |
|---|---|
| `pip install batcher-engine` | Every runnable block on this page |
| A Kafka broker | Only the final block, which is shown and not run |

## 1. A stream

An unbounded source is any function yielding Arrow batches. `bounded=False` is what tells
the engine this thing never ends.

```python
import datetime as dt

import batcher as bt
import pyarrow as pa

schema = pa.schema([("user", pa.string()), ("ts", pa.timestamp("us")), ("amount", pa.int64())])
start = dt.datetime(2024, 5, 1, 9, 0)


def feed():
    yield pa.record_batch(
        {
            "user": ["a", "b"],
            "ts": [start, start + dt.timedelta(minutes=10)],
            "amount": [10, 5],
        },
        schema=schema,
    )
    yield pa.record_batch(
        {
            "user": ["a", "c"],
            "ts": [start + dt.timedelta(minutes=70), start + dt.timedelta(minutes=80)],
            "amount": [7, 3],
        },
        schema=schema,
    )


events = bt.from_batches(feed, schema, bounded=False)
print(events.is_streaming)
# True
```

In production the source is `bt.read.kafka(...)`, `bt.read.kinesis(...)`,
`bt.read.delta(uri, stream=True)`, or `bt.read.files_incremental(...)`. Nothing below this
line changes when you swap it.

## 2. Transform it exactly like a table

There is no streaming dialect. `filter`, `select`, `with_columns`, `group_by`, `join` are the
same operators.

```python
big = events.filter(bt.col("amount") > 4)
print(sum(batch.num_rows for batch in big.iter_batches()))
# 3
```

:::{warning}
An unbounded dataset cannot `collect()`. It would never finish, and it raises a clear
`PlanError` if you try. Consume it with `iter_batches()` or write it to a sink. This is the
first thing everyone hits, and the error message is telling you the truth rather than being
awkward.
:::

## 3. Deduplicate, in bounded memory

Streams redeliver. `drop_duplicates_within_watermark` keeps the first row per key seen
inside the watermark window and *forgets* keys the watermark has passed, so its state cannot
grow without bound.

```python
deduped = bt.from_batches(feed, schema, bounded=False).drop_duplicates_within_watermark(
    ["user"], event_time="ts", lateness="1h"
)
seen = [u for batch in deduped.iter_batches() for u in batch.column("user").to_pylist()]
print(sorted(seen))
# ['a', 'b', 'c']
```

User `a` appears twice in the feed and once in the output.

## 4. Window by event time

`bt.window(time_col, duration)` assigns each row to an event-time window; group by it like
any other key. Event time is the timestamp *in the row*, not the clock on the machine, which
is the only definition that survives a replay.

The watermark (`max(event_time) - lateness`) is what lets the engine emit a window and evict
its state: once the watermark passes a window's end, that window is closed and its memory is
released. Rows later than that are dropped rather than reopening it.

```python
hourly = (
    bt.from_batches(feed, schema, bounded=False)
    .with_watermark("ts", "15m")
    .group_by(w=bt.window(bt.col("ts"), "1h"))
    .agg(revenue=bt.col("amount").sum())
)
```

Nothing has run yet. It is still a lazy plan.

## 5. Write it, with a trigger

`ds.write` is the one write surface. Give it a `trigger` and it runs as a streaming query,
appending each micro-batch and handing you back a `StreamingQuery`.

`Trigger.available_now()` drains everything the source has at that moment and stops, which is
the incremental-batch and backfill cadence, and the one that makes a tutorial finish.
`Trigger.processing_time("30 seconds")` is the continuous one.

| Choice | Emits | Use it for |
|---|---|---|
| `Trigger.available_now()` | Everything available, then stops | Backfills, incremental batch, and tutorials that need to end |
| `Trigger.processing_time("30 seconds")` | A micro-batch on a clock | A continuous query |
| `output_mode="append"` (default) | Only rows that are final and will never change | An event log, a bronze layer |
| `output_mode="complete"` | The whole result table, every micro-batch | A running-totals view |

```python
query = hourly.write.memory(
    "hourly_revenue",
    trigger=bt.Trigger.available_now(),
    output_mode="complete",
)
query.await_termination()

print(bt.read_memory("hourly_revenue").sort("w").to_pydict())
# {'w': [datetime.datetime(2024, 5, 1, 9, 0), datetime.datetime(2024, 5, 1, 10, 0)], 'revenue': [15, 10]}
```

Two windows: 09:00 holds `10 + 5`, 10:00 holds `7 + 3`. `output_mode="complete"` re-emits
the whole result table each micro-batch, which is what a running-totals view wants;
`"append"` (the default) emits only rows that are final and will never change again.

## 6. Land it in files, and survive a restart

A file sink writes one part file per micro-batch. Pass `checkpoint=` and the query records
its source offsets and sink commits, so a restart resumes at the last committed offset
instead of reprocessing from the beginning.

```python
import os
import tempfile

work = tempfile.mkdtemp()
bronze = os.path.join(work, "bronze")

q = (
    bt.from_batches(feed, schema, bounded=False)
    .filter(bt.col("amount") > 4)
    .write(
        bronze,
        format="parquet",
        trigger=bt.Trigger.available_now(),
        checkpoint=os.path.join(work, "_checkpoint"),
        query_name="bronze_ingest",
    )
)
q.await_termination()

print(bt.read.parquet(bronze).count())
# 3
print(q.is_active, q.exception())
# False None
```

:::{important}
Give the query a stable `query_name`. Writing to Delta, it becomes the transaction id the
sink checks the log for, and a replayed micro-batch that finds its own transaction already
committed writes nothing. That is what turns at-least-once replay into end-to-end
exactly-once. Change the name between restarts and the sink has no way to recognize a
replay, so the batch is written twice and the duplicate is yours to find.
:::

## 7. Custom per-batch logic

`for_each_batch` hands you the whole Arrow table for each micro-batch, never a row. It is the
hook for a custom upsert, a fan-out to several sinks, or a commit protocol the built-in sinks
do not cover.

```python
batches = []
q2 = bt.from_batches(feed, schema, bounded=False).write.for_each_batch(
    lambda table, batch_id: batches.append((batch_id, table.num_rows)),
    trigger=bt.Trigger.available_now(),
)
q2.await_termination()
print(batches)
# [(0, 2), (1, 2)]
```

## 8. The real thing

Swap the generator for Kafka, the memory sink for Delta, and `available_now` for a
processing-time trigger. The query in the middle is untouched.

```python
# docs: skip
import batcher as bt

(
    bt.read.kafka(topic="orders", bootstrap_servers="localhost:9092")
    .with_watermark("ts", "15m")
    .group_by(w=bt.window(bt.col("ts"), "1h"))
    .agg(revenue=bt.col("amount").sum())
    .write.delta(
        "s3://lake/gold/hourly_revenue",
        trigger=bt.Trigger.processing_time("1 minute"),
        output_mode="append",
        checkpoint="s3://lake/gold/_checkpoint",
        query_name="hourly_revenue",
    )
)
```

Manage it with the handle: `q.status`, `q.recent_progress()`, `q.stop()`, and
`bt.streams()` for every active query in the process.

## What you learned

- One API. The operators do not know whether the source ends.
- A watermark is not optional bookkeeping. It is what bounds the memory of a stateful
  streaming operator, and what makes late data a decision rather than a leak.
- A checkpoint plus a stable `query_name` is the whole exactly-once story.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming guide
:link: /user-guide/moving-data/streaming
:link-type: doc
Every source, sink, trigger, and output mode.
:::

:::{grid-item-card} {octicon}`database;1.1em` Building a lakehouse
:link: /tutorials/pipelines/building-a-lakehouse
:link-type: doc
The medallion layers this pipeline feeds.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: /user-guide/analyze/window-functions
:link-type: doc
The SQL `OVER` family, on bounded and unbounded data alike.
:::
::::

## See also

- {doc}`Kafka integration </integrations/streams/kafka>`: the source the generator stands in for.
- {doc}`Windowed aggregation </cookbook/streaming/windowed-aggregation>` and
  {doc}`exactly-once sink </cookbook/streaming/exactly-once-sink>`: the recipes for steps 4
  through 6.
- {doc}`Late data and watermarks </cookbook/streaming/late-data-watermarks>`: what happens to
  a row that arrives after its window closed.
- {doc}`Deduplication </cookbook/data-engineering/maintenance/deduplication>`: the bounded-memory dedup,
  in the batch case.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: what a checkpoint actually
  guarantees.
