# A streaming pipeline

Build a continuous pipeline: read an unbounded stream, dedupe it, window it by event time, and write each micro-batch out. Batch is the bounded special case of streaming, so the operators are the ones you already know. Only the source and the trigger change. A generator stands in for Kafka, so everything runs as written.

| You need | For |
|---|---|
| `pip install batcher-engine` | Every runnable block on this page |
| A Kafka broker | Only the final block, which is shown and not run |

Each step below adds one stage, and the numbers in the diagram are the step numbers on this page:

![Five stages run top to bottom. Step 1 is an unbounded source built with from_batches and bounded=False, which never ends, so you consume it with a sink or iter_batches rather than collect(). It passes Arrow batches to step 3, drop_duplicates_within_watermark, which keeps the first row per key and forgets keys the watermark has passed, so its state stays bounded. Step 4 sets with_watermark and groups by bt.window. The watermark is max(event_time) minus lateness, and a window closes once the watermark passes its end. The window aggregates go to step 5, a write with a trigger, where available_now drains what is there and stops and processing_time runs on a clock. Its micro-batches reach step 6, a memory, Parquet, or Delta sink with a checkpoint, which resumes at the last committed offset after a restart.](/_static/diagrams/streaming_tutorial_flow.svg)

## 1. A stream

An unbounded source is any function yielding Arrow batches, marked with `bounded=False`:

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

An in-memory table is the bounded case:

```python
print(bt.from_pydict({"amount": [1]}).is_streaming)
# False
```

In production the source is {py:meth}`bt.read.kafka(...) <batcher.api.io_namespace.reader.Reader.kafka>`, {py:meth}`bt.read.kinesis(...) <batcher.api.io_namespace.reader.Reader.kinesis>`,
{py:meth}`bt.read.delta(uri, stream=True) <batcher.api.io_namespace.reader.Reader.delta>`, or {py:meth}`bt.read.files_incremental(...) <batcher.api.io_namespace.reader.Reader.files_incremental>`. Nothing below this
line changes when you swap it in.

## 2. Transform it exactly like a table

There is no streaming dialect. `filter`, `select`, `with_columns`, `group_by`, and `join` work as they do on a table:

```python
big = events.filter(bt.col("amount") > 4)
print(sum(batch.num_rows for batch in big.iter_batches()))
# 3
```

Consume a stream with {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` or a sink, or peek at it with `limit(n)`:

```python
print(events.limit(3).to_pydict()["user"])
# ['a', 'b', 'a']
```

{py:meth}`collect() <batcher.Dataset.collect>` on an unbounded dataset raises {py:exc}`PlanError <batcher.PlanError>`, since it would never finish.

## 3. Deduplicate, in bounded memory

{py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` keeps the first row per key inside the watermark window and forgets keys the watermark has passed, so its state stays bounded:

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

`bt.window(time_col, duration)` assigns each row to an event-time window, using the timestamp *in the row*, so a replay gives the same answer. Group by it like any other key. The watermark (`max(event_time) - lateness`) closes a window once it passes the window's end, which emits the window and frees its state.

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

Give {py:obj}`ds.write <batcher.Dataset.write>` a `trigger` and it runs as a streaming query that returns a `StreamingQuery` handle. {py:meth}`Trigger.available_now() <batcher.Trigger.available_now>` drains what is there and stops. {py:meth}`Trigger.processing_time("30 seconds") <batcher.Trigger.processing_time>` runs continuously.

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

Two windows: 09:00 holds `10 + 5`, 10:00 holds `7 + 3`.

## 6. Land it in files, and survive a restart

A file sink writes one part file per micro-batch. `checkpoint=` records source offsets and sink commits, so a restart resumes at the last committed offset.

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
Keep `query_name` stable across restarts. On a Delta sink it becomes the transaction id that makes a replayed micro-batch a no-op, which gives end-to-end exactly-once.
:::

## 7. Custom per-batch logic

{py:meth}`for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>` hands you each micro-batch as a whole Arrow table, for a custom upsert or a fan-out to several sinks:

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

Manage it with the handle: `q.status`, `q.recent_progress`, `q.stop()`, and
{py:func}`bt.streams() <batcher.streams>` for every active query in the process.

## Where to go next

Settle the watermark first. It bounds the memory of every stateful streaming operator and decides what counts as late.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming guide
:link: /user-guide/moving-data/streaming/index
:link-type: doc
Every source, sink, trigger, and output mode.
:::

:::{grid-item-card} {octicon}`database;1.1em` Building a lakehouse
:link: /getting-started/tutorials/pipelines/building-a-lakehouse
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
