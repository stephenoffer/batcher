# Streaming

Batcher treats **batch as the bounded special case of streaming**. The same
`Dataset` API — the same `group_by`, `window`, `join`, `with_columns`, `write` —
runs over a finite table or an unbounded stream. You do not learn a second API or
rewrite a pipeline to move it from a one-off job to a continuous one: you change the
*source* (or add a `trigger`), and the same code keeps working.

This page covers consuming and producing streams: sources, the unified `ds.write`,
triggers and output modes, event-time windows and watermarks, deduplication,
stream–stream joins, and exactly-once checkpointing.

## One API, batch or streaming

Every operation below works on a bounded dataset (returning a result you can
`collect()`) and on an unbounded one (consumed with `iter_batches()` or written to a
sink). The query is identical:

```python
import batcher as bt
from batcher import col

events = bt.from_pydict({"user": ["a", "b", "a", "c"], "amount": [10, 5, 7, 3]})

# Bounded source → a finite result.
totals = events.group_by("user").agg(total=col("amount").sum())
print(totals.to_pydict())
```

Point the *same* transformation at an unbounded source and consume it incrementally
— nothing about the pipeline changes:

```python
import pyarrow as pa

schema = pa.schema([("user", pa.string()), ("amount", pa.int64())])

def feed():
    yield pa.record_batch({"user": ["a", "b"], "amount": [10, 5]}, schema=schema)
    yield pa.record_batch({"user": ["a", "c"], "amount": [7, 3]}, schema=schema)

stream = bt.from_batches(feed, schema, bounded=False)
# The bounded-memory streaming path is chosen automatically.
seen = [b.num_rows for b in stream.filter(col("amount") > 4).iter_batches()]
print(sum(seen))
```

A bounded source can additionally `collect()`; an unbounded one cannot (it would
never finish) and raises a clear `PlanError` if a terminal tries to materialize it.
Use `ds.is_streaming` to check which you have.

## Reading streams

Streaming sources are unbounded relations behind the same `bt.read` namespace as
files and tables:

| Source | Reader |
| --- | --- |
| Apache Kafka | `bt.read.kafka(topic="events", ...)` |
| Amazon Kinesis | `bt.read.kinesis(stream_name="...")` |
| Incremental files (Auto Loader) | `bt.read.files_incremental(path, "parquet", state_dir=...)` |
| Delta Lake (new commits) | `bt.read.delta(uri, stream=True)` |
| Delta Change Data Feed | `bt.read.read_change_feed(uri)` |
| Rate generator (dev) | `bt.read.rate(rows_per_second)` |
| TCP socket (dev) | `bt.read.socket(host, port)` |

The `rate` source generates rows and is handy for trying the API without external
infrastructure (`num_rows` bounds it, `pace=False` removes the one-second cadence):

```python
demo = bt.read.rate(5, num_rows=10, pace=False)
rows = [b.num_rows for b in demo.iter_batches()]
print(sum(rows))  # 10 generated (value, timestamp) rows
```

Kafka, Kinesis, and Delta need their optional dependency and a running service:

```python
# docs: skip
clicks = bt.read.kafka(topic="clicks", bootstrap_servers="localhost:9092")
recent = clicks.filter(col("partition") == 0)
for batch in recent.iter_batches():
    handle(batch)
```

## Writing streams: the unified `ds.write`

`ds.write(...)` is the one write surface. With a bounded source and no trigger it is
a single batch write returning a `WriteManifest` (unchanged from a normal job). Add a
`trigger=` — or point it at an unbounded source — and it runs as a streaming query,
appending each micro-batch and returning a `StreamingQuery` handle:

```python
import pyarrow as pa

schema = pa.schema([("user", pa.string()), ("amount", pa.int64())])

def feed():
    yield pa.record_batch({"user": ["a", "b"], "amount": [10, 5]}, schema=schema)
    yield pa.record_batch({"user": ["a"], "amount": [7]}, schema=schema)

stream = bt.from_batches(feed, schema, bounded=False)

query = stream.write.memory("totals_demo", trigger=bt.Trigger.available_now())
query.await_termination()
print(bt.read_memory("totals_demo").count())  # 3 rows accumulated
```

Sinks available on the write namespace:

- `ds.write(path, format=..., trigger=...)` — files (Parquet/CSV/JSON/…), one
  `part-batch*` file per micro-batch, idempotent on restart.
- `ds.write.delta(uri, trigger=...)` — a transactional Delta append per micro-batch.
- `ds.write.console(trigger=...)` — print each micro-batch (development).
- `ds.write.memory(name, trigger=...)` — an in-memory table read back with
  `bt.read_memory(name)`.
- `ds.write.for_each_batch(fn, trigger=...)` — call `fn(table, batch_id)` on each
  micro-batch. The whole Arrow table is passed (never a row), so this is the hook for
  custom upserts (`MERGE`/SCD), multi-sink fan-out, or any per-batch commit logic.
- `ds.write.for_each(fn, trigger=...)` — call `fn(row)` per row.

### Triggers

A `Trigger` sets the cadence (Spark parity):

- `bt.Trigger.processing_time("5 seconds")` — fire a micro-batch on a wall-clock
  interval (the default streaming cadence).
- `bt.Trigger.once()` — process one micro-batch of available data, then stop.
- `bt.Trigger.available_now()` — drain all currently-available data, then stop (the
  incremental-batch / backfill trigger).
- `bt.Trigger.continuous("1 second")` — lowest latency: micro-batches run
  back-to-back with no inter-batch delay, committing a checkpoint epoch on the
  interval (stateless pipelines only).

### Output modes

`output_mode=` controls what each micro-batch emits:

- `"append"` (default) — only rows that are final and will not change again. For a
  plain pipeline that is every row; for a windowed aggregation it is a window's row
  once the watermark closes it.
- `"complete"` — the full result table after every micro-batch (aggregations only).
- `"update"` — only the result rows whose value changed this micro-batch.

```python
agg_stream = bt.from_batches(feed, schema, bounded=False).group_by("user").agg(
    total=col("amount").sum()
)
q = agg_stream.write.memory("running_totals", trigger=bt.Trigger.available_now(),
                            output_mode="complete")
q.await_termination()
print(dict(zip(*[bt.read_memory("running_totals").to_pydict()[c]
                 for c in ("user", "total")], strict=True)))
```

### Managing a query

`start`-style writes return a `StreamingQuery`:

```python
# docs: skip
q = clicks.write("s3://bucket/out", format="parquet",
                 trigger=bt.Trigger.processing_time("10 seconds"),
                 checkpoint="s3://bucket/_ckpt")
q.is_active            # True while running
q.status               # a point-in-time StreamingQueryStatus
q.recent_progress()    # per-micro-batch metrics
q.stop()               # halt at the next micro-batch boundary
bt.streams()           # all active streaming queries
```

## Event-time windows and watermarks

`bt.window(time_col, duration[, slide])` assigns each row to an event-time window;
group by it like any other key. Tumbling (no `slide`) and sliding windows both work,
batch or streaming:

```python
import datetime as dt

base = dt.datetime(2024, 1, 1)
clicks = bt.from_pydict({
    "ts": [base, base + dt.timedelta(minutes=30), base + dt.timedelta(minutes=90)],
    "n": [1, 2, 3],
})
hourly = clicks.group_by(w=bt.window(col("ts"), "1h")).agg(hits=col("n").sum())
print(hourly.to_pydict())  # 00:00 → 3, 01:00 → 3
```

On an unbounded stream, declare a **watermark** so windowed state stays bounded:
`ds.with_watermark(time_col, lateness)` lets the engine emit and evict a window once
the watermark (`max(event_time) - lateness`) passes its end, and drop rows that
arrive later than that. The query is otherwise identical to the batch one:

```python
# docs: skip
windowed = (
    bt.read.kafka(topic="clicks")
    .with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1 hour"))
    .agg(hits=col("n").sum())
)
windowed.write.delta("gold/hourly", trigger=bt.Trigger.processing_time("1 minute"),
                     output_mode="append", checkpoint="gold/_ckpt")
```

**Session windows** group consecutive events whose gap is below a timeout:

```python
sessions = clicks.session_window("ts", "45m", hits=col("n").sum())
print(sessions.select("session_start", "session_end", "hits").to_pydict())
```

## Deduplication within a watermark

`drop_duplicates_within_watermark` keeps the first row per key seen inside the
watermark window, forgetting keys the watermark has passed so memory stays bounded.
Over a bounded source it is exact deduplication:

```python
records = bt.from_pydict({
    "id": ["x", "y", "x", "z"],
    "ts": [base, base, base + dt.timedelta(minutes=1), base],
    "v": [1, 2, 3, 4],
})
deduped = records.drop_duplicates_within_watermark(["id"], event_time="ts",
                                                   lateness="1h")
print(sorted(deduped.to_pydict()["id"]))  # ['x', 'y', 'z'] — the second 'x' dropped
```

## Stream–stream joins

`join_stream` joins two streams on keys **and** an event-time interval
(`|left_time - right_time| <= within`). The time bound is what lets buffered state be
evicted by the watermark, keeping a two-stream join in bounded memory. Bounded
sources run it as a plain join plus the interval filter:

```python
impressions = bt.from_pydict({"ad": ["a", "b"], "shown": [base, base]})
clicks2 = bt.from_pydict({"ad": ["a"], "clicked": [base + dt.timedelta(minutes=2)]})

attributed = impressions.join_stream(
    clicks2, on="ad", left_time="shown", right_time="clicked", within="5m"
)
print(attributed.to_pydict()["ad"])  # ['a'] — clicked within 5 minutes of shown
```

## Exactly-once and checkpointing

Pass `checkpoint=<dir>` to a streaming write to record source offsets and sink
commits per micro-batch. On restart the query resumes from the last committed offset
— a replayable source (Kafka offsets, Kinesis sequence numbers, a Delta version,
the Auto-Loader seen-file set) seeks forward and an idempotent sink dedups, so the
combined output is exactly-once with no row lost or duplicated:

```python
# docs: skip
q = bt.read.kafka(topic="orders").write(
    "lake/bronze", format="parquet",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="lake/bronze/_checkpoint",
)
# A crash and restart of the same query against the same checkpoint resumes
# exactly where it left off.
```

## The medallion pattern

Because each layer reads the previous one as a stream, the three medallion layers
chain with the same primitives — incremental file or Delta reads in, transform,
checkpointed write out:

```python
# docs: skip
# Bronze: raw ingestion.
bt.read.kafka(topic="events").write(
    "lake/bronze", format="parquet",
    trigger=bt.Trigger.available_now(), checkpoint="lake/bronze/_ck")

# Silver: clean + dedup, reading bronze incrementally.
(bt.read.files_incremental("lake/bronze", "parquet", state_dir="lake/silver/_seen")
   .drop_duplicates_within_watermark(["id"], event_time="ts", lateness="10m")
   .write("lake/silver", format="parquet",
          trigger=bt.Trigger.available_now(), checkpoint="lake/silver/_ck"))

# Gold: windowed aggregates, reading silver incrementally.
(bt.read.files_incremental("lake/silver", "parquet", state_dir="lake/gold/_seen")
   .with_watermark("ts", "10m")
   .group_by(w=bt.window(col("ts"), "1h"))
   .agg(total=col("v").sum())
   .write.delta("lake/gold", trigger=bt.Trigger.available_now(),
                output_mode="append", checkpoint="lake/gold/_ck"))
```

## See also

- {doc}`writing-data` — the batch write surface `ds.write` extends.
- {doc}`aggregations` and {doc}`window-functions` — the grouping and SQL-window APIs.
- {doc}`../architecture/execution` — the pipelines-and-breakers execution model that
  makes batch and streaming one engine.
- {doc}`../ml/streaming` — streaming a query as bounded-memory training data.
```
