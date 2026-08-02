# Kafka to the lake

The pipeline is the easy part: decode the payload, drop the junk, land it in Parquet.
What makes it a streaming job is the restart.

:::{important}
Batcher records a micro-batch's source offset *before* it processes the batch, so a crash in
between leaves a batch the next run will read again. A sink that only appends writes those
rows twice. The engine is at-least-once by design, and end-to-end exactly-once is bought by
the sink, not by the source. Everything below is arranged around that one fact.
:::

## What a broker gives you

Every broker source (Kafka, Kinesis, Pulsar, Pub/Sub, Event Hubs) hands you the same
six columns, and the payload is opaque bytes:

| column | type | |
| --- | --- | --- |
| `key` | binary | may be null |
| `value` | binary | your payload, undecoded |
| `partition` | int64 | topic-partition / shard |
| `offset` | int64 | position within the partition |
| `timestamp` | int64 | milliseconds since the epoch; *broker* time, not event time |
| `topic` | string | |

Decoding is your job, and it is an ordinary expression over `value`. That means you can
build and test the whole ETL on a batch of rows shaped like a poll, with no broker
running. This block is the real thing, executed when the docs are tested:

```python
import json

import pyarrow as pa

import batcher as bt
from batcher import col

broker = pa.schema([
    ("key", pa.binary()),
    ("value", pa.binary()),
    ("partition", pa.int64()),
    ("offset", pa.int64()),
    ("timestamp", pa.int64()),
    ("topic", pa.string()),
])

payloads = [
    {"user": "u1", "event": "click", "amount": 3},
    {"user": "u2", "event": "view", "amount": 0},
    {"user": "u1", "event": "click", "amount": 5},
]
poll = pa.record_batch(
    {
        "key": [p["user"].encode() for p in payloads],
        "value": [json.dumps(p).encode() for p in payloads],
        "partition": [0, 0, 1],
        "offset": [10, 11, 4],
        "timestamp": [1704067200000, 1704067201000, 1704067202000],
        "topic": ["clicks"] * 3,
    },
    schema=broker,
)
raw = bt.from_arrow(pa.Table.from_batches([poll]))
```

The decode: cast the bytes to a string, pull the fields out with JSON paths, keep the
broker coordinates you care about. No Python per row: these are `Expr`s, and they run
in Rust.

```python
def decode(ds):
    return ds.with_columns(payload=col("value").cast("string")).select(
        user=col("payload").json.extract_string("$.user"),
        event=col("payload").json.extract_string("$.event"),
        amount=col("payload").json.extract_int("$.amount"),
        offset=col("offset"),
        partition=col("partition"),
    )
```

## The same `decode`, bounded and unbounded

The transformation is one function. What changes is the source under it and the terminal on
top of it.

::::{tab-set}
:::{tab-item} A poll-shaped fixture

Bounded, so it collects, and so the docs test can run it.

```python
clicks = decode(raw).filter(col("event") == "click")
print(clicks.to_pydict())
# {'user': ['u1', 'u1'], 'event': ['click', 'click'], 'amount': [3, 5], 'offset': [10, 4], 'partition': [0, 1]}
```
:::

:::{tab-item} A live topic

Point it at Kafka and nothing in the transformation changes. The source becomes
unbounded, so `collect()` is refused and the terminal is a streaming write:

```python
# docs: skip
raw = bt.read.kafka("clicks", bootstrap_servers="broker-1:9092", group="bronze-etl")

query = (
    decode(raw)
    .filter(col("event") == "click")
    .write(
        "s3://lake/bronze/clicks",
        format="parquet",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="s3://lake/bronze/clicks/_ckpt",
        query_name="bronze-clicks",
    )
)
query.await_termination()
```
:::
::::

Three arguments are doing the work. `trigger` sets the micro-batch cadence.
`checkpoint` is where the offset and commit logs live. Without it, a restart cannot know
what it already read. `query_name` is stable identity: it is the transaction id a
Delta sink uses for its idempotency check, so it must not change across restarts.

The cadence is the one you will change most often:

| Trigger | Cadence | Reach for it when |
| --- | --- | --- |
| `bt.Trigger.processing_time("30 seconds")` | a micro-batch on a wall-clock interval | the job is always on; this is the default cadence |
| `bt.Trigger.available_now()` | drains everything available, then stops | the job is a cron-style incremental batch, or a backfill |
| `bt.Trigger.once()` | one micro-batch of available data, then stops | you want a single step, for a test or a manual catch-up |
| `bt.Trigger.continuous("1 second")` | back-to-back micro-batches, checkpointing on the interval | latency is the whole point, and the pipeline is stateless |

:::{dropdown} How the Kafka source behaves: offsets, splits, and that `timestamp` column

The consumer commits offsets to the group
*after* a batch is assembled, so a crash before that commit re-delivers those messages.
At-least-once, by design. `splits()` returns one split per topic-partition, so a
distributed reader assigns one consumer per partition. And `timestamp` is when the
broker got the message; if you need event time, it is a field inside your payload, and
you should extract it explicitly.
:::

## Prove the restart behavior without a broker

You do not need Kafka to exercise the "what happens on the second run" question. The
incremental file source (`files_incremental`, the Auto Loader analog) is unbounded in
exactly the same way, and it keeps a durable seen-file store in `state_dir`. Drop a file
in, drain it, drop another file in, drain again. The second run only sees the new file:

```python
import os
import tempfile

import pyarrow.parquet as pq

root = tempfile.mkdtemp()
inbox = os.path.join(root, "inbox")
os.makedirs(inbox)
seen = os.path.join(root, "_seen")

pq.write_table(pa.table({"user": ["u1", "u2"], "amount": [3, 5]}), f"{inbox}/001.parquet")

first = bt.read.files_incremental(inbox, "parquet", state_dir=seen)
q = first.filter(col("amount") > 2).write.memory(
    "bronze_pass1", trigger=bt.Trigger.available_now()
)
q.await_termination()
print(bt.read_memory("bronze_pass1").to_pydict())
# {'user': ['u1', 'u2'], 'amount': [3, 5]}
```

A new file arrives. Re-run the identical job:

```python
pq.write_table(pa.table({"user": ["u3"], "amount": [9]}), f"{inbox}/002.parquet")

second = bt.read.files_incremental(inbox, "parquet", state_dir=seen)
q = second.filter(col("amount") > 2).write.memory(
    "bronze_pass2", trigger=bt.Trigger.available_now()
)
q.await_termination()
print(bt.read_memory("bronze_pass2").to_pydict())
# {'user': ['u3'], 'amount': [9]}
```

`Trigger.available_now()` drains what is there and stops. That is the trigger you want
for a cron-style incremental batch: same code as the always-on job, different cadence.

## Rough edges, stated plainly

:::{warning}
The incremental file source tracks files in its own `state_dir`, not in the streaming
`checkpoint` offset log. They are two different pieces of state. Delete `state_dir`
and you re-ingest the directory.
:::

:::{warning}
`distributed=True` on a streaming write fans the read across the cluster, but only for
an `available_now`/`once` drain, only for a stateless pipeline, and it does **not**
checkpoint. Backfills, not the steady-state job.
:::

Landing raw bytes in bronze and decoding downstream is often the better call: a schema
mistake in `decode` is then a replay of your own Parquet, not of Kafka's retention
window.

## See also

- {doc}`Exactly-once sinks </cookbook/streaming/exactly-once-sink>`: what the checkpoint actually guarantees,
  and the ways a file sink can silently drop data.
- {doc}`Windowed aggregation </cookbook/streaming/windowed-aggregation>`: the gold layer over this bronze one.
- {doc}`Streaming inference </cookbook/streaming/streaming-inference>`: scoring these events as they land.
- {doc}`Streaming </user-guide/moving-data/streaming>`: the full source/sink/trigger reference.
- {doc}`Kafka integration </integrations/streams/kafka>`: consumer groups, splits, and the broker
  schema above.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: the sink surface and Delta commits.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: `files_incremental` and the rest of the
  sources.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: the same directory-watching
  job, run as a batch.
