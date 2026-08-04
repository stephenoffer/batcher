# Streaming

Batcher treats **batch as the bounded special case of streaming**. One {py:class}`Dataset <batcher.Dataset>`
API (`group_by`, `window`, `join`, `with_columns`, `write`) runs over a finite table
or an unbounded stream. Moving a pipeline from a one-off job to a continuous one
means changing the *source*, or adding a `trigger`. There is no second API to learn
and nothing in the pipeline to rewrite.

## One API, batch or streaming

Every operation below works on a bounded dataset (returning a result you can
{py:meth}`collect() <batcher.Dataset.collect>`) and on an unbounded one (consumed with {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` or written to a
sink). The query is identical:

```python
import batcher as bt
from batcher import col

events = bt.from_pydict({"user": ["a", "b", "a", "c"], "amount": [10, 5, 7, 3]})

# Bounded source → a finite result.
totals = events.group_by("user").agg(total=col("amount").sum())
print(totals.to_pydict())
```

Point that same transformation at an unbounded source and consume it incrementally.
Nothing about the pipeline changes.

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

A bounded source can also `collect()`. An unbounded one cannot, since it would never
finish, and it raises a clear {py:exc}`PlanError <batcher.PlanError>` if a terminal tries to materialize it. Use
{py:obj}`ds.is_streaming <batcher.Dataset.is_streaming>` to check which you have.

## Reading streams

Streaming sources are unbounded relations behind the same {py:obj}`bt.read <batcher.read>` namespace as
files and tables:

| Source | Reader |
| --- | --- |
| Apache Kafka | {py:meth}`bt.read.kafka("events", bootstrap_servers=...) <batcher.api.io_namespace.reader.Reader.kafka>` |
| Amazon Kinesis | {py:meth}`bt.read.kinesis("my-stream", region=...) <batcher.api.io_namespace.reader.Reader.kinesis>` |
| Apache Pulsar | {py:meth}`bt.read.pulsar("events", service_url=...) <batcher.api.io_namespace.reader.Reader.pulsar>` |
| Google Pub/Sub | {py:meth}`bt.read.pubsub("projects/p/subscriptions/s") <batcher.api.io_namespace.reader.Reader.pubsub>` |
| Azure Event Hubs | {py:meth}`bt.read.eventhubs("hub", connection_str=...) <batcher.api.io_namespace.reader.Reader.eventhubs>` |
| Incremental files (Auto Loader) | {py:meth}`bt.read.files_incremental(path, "parquet", state_dir=...) <batcher.api.io_namespace.reader.Reader.files_incremental>` |
| Delta Lake (new commits) | {py:meth}`bt.read.delta(uri, stream=True) <batcher.api.io_namespace.reader.Reader.delta>` |
| Delta Change Data Feed | {py:meth}`bt.read.read_change_feed(uri) <batcher.api.io_namespace.reader.Reader.read_change_feed>` |
| Rate generator (dev) | {py:meth}`bt.read.rate(rows_per_second) <batcher.api.io_namespace.reader.Reader.rate>` |
| TCP socket (dev) | {py:meth}`bt.read.socket(host, port) <batcher.api.io_namespace.reader.Reader.socket>` |

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

{py:obj}`ds.write(...) <batcher.Dataset.write>` is the one write surface. With a bounded source and no trigger it is
a single batch write returning a {py:class}`WriteManifest <batcher.io.WriteManifest>`, unchanged from a normal job. Add a
`trigger=`, or point it at an unbounded source, and it runs as a streaming query:
each micro-batch is appended, and you get back a `StreamingQuery` handle.

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

- `ds.write(path, format=..., trigger=...)` writes files such as Parquet, CSV, or JSON,
  one `part-batch*` file per micro-batch, idempotent on restart.
- {py:meth}`ds.write.delta(uri, trigger=...) <batcher.api.io_namespace.writer.Writer.delta>` makes a transactional Delta append per micro-batch.
- {py:meth}`ds.write.console(trigger=..., num_rows=..., truncate=...) <batcher.api.io_namespace.writer.Writer.console>` prints each micro-batch.
  Development only. Strings are shortened to 20 characters for display unless you pass
  `truncate=False` or an explicit width.
- {py:meth}`ds.write.memory(name, trigger=...) <batcher.api.io_namespace.writer.Writer.memory>` builds an in-memory table you read back with
  {py:func}`bt.read_memory(name) <batcher.read_memory>`.
- {py:meth}`ds.write.for_each_batch(fn, trigger=...) <batcher.api.io_namespace.writer.Writer.for_each_batch>` calls `fn(table, batch_id)` on each
  micro-batch. The whole Arrow table is passed, never a row, so this is the hook for
  custom upserts (`MERGE`/SCD), multi-sink fan-out, or any per-batch commit logic.
- {py:meth}`ds.write.for_each(fn, trigger=...) <batcher.api.io_namespace.writer.Writer.for_each>` calls `fn(row)` per row. Pass a
  {py:class}`ForeachWriter <batcher.ForeachWriter>` instead of a function when the
  destination needs a connection: its `open(partition_id, epoch_id)` acquires one and
  returns whether to proceed, `process(row)` writes a row, and `close(error)` releases it,
  including when the epoch failed. A bare function has nowhere to put a connection.
- {py:meth}`ds.write.kafka(topic, bootstrap_servers=..., trigger=...) <batcher.api.io_namespace.writer.Writer.kafka>` publishes each row to Kafka.
  The column contract is Spark's: `value` is required, and `key`, `topic`, `partition`, and
  `headers` are optional. Delivery is at-least-once, so make the consumer idempotent or
  dedup on the key.

### Triggers

A {py:class}`Trigger <batcher.Trigger>` sets the cadence (Spark parity):

- {py:meth}`bt.Trigger.processing_time("5 seconds") <batcher.Trigger.processing_time>` fires a micro-batch on a wall-clock
  interval. This is the default streaming cadence.
- {py:meth}`bt.Trigger.once() <batcher.Trigger.once>` processes one micro-batch of available data, then stops.
- {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>` drains every record available when it starts, then stops.
  It is the incremental-batch and backfill trigger.
- {py:meth}`bt.Trigger.continuous("1 second") <batcher.Trigger.continuous>` is the lowest-latency option: micro-batches run
  back-to-back with no inter-batch delay, committing a checkpoint epoch on the
  interval. Stateless pipelines only.

### Output modes

`output_mode=` controls what each micro-batch emits:

- `"append"` (default): only rows that are final and will not change again. For a
  plain pipeline that is every row; for a windowed aggregation it is a window's row
  once the watermark closes it.
- `"complete"`: the full result table after every micro-batch. Aggregations only.
- `"update"`: only the result rows whose value changed this micro-batch.

Those literals are the values of the {py:class}`OutputMode <batcher.OutputMode>`
constants, {py:obj}`bt.OutputMode.APPEND <batcher.OutputMode.APPEND>`, {py:obj}`bt.OutputMode.COMPLETE <batcher.OutputMode.COMPLETE>`, {py:obj}`bt.OutputMode.UPDATE <batcher.OutputMode.UPDATE>`.
Pass a constant in place of the raw string for a typo-proof spelling.

```python
print(bt.OutputMode.COMPLETE)
# complete
```

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

## Monitoring a running query

A streaming write hands back a `StreamingQuery`. It carries the query's liveness, the
per-micro-batch metrics, the state each stateful operator holds, and the count of rows
dropped as late. See {doc}`streaming-monitoring`.

## Event-time windows and watermarks

`bt.window(time_col, duration)` assigns each row to one event-time window. Group by it like
any other key, batch or streaming:

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

### Sliding windows explode, they do not group

Pass a third argument and the windows overlap: `bt.window(col("ts"), "1h", "30m")` is a
one-hour window advancing every thirty minutes, so a single row belongs to *two* of them.
The expression therefore evaluates to the **list** of starts that contain the row, not to
one start. Fan that list out with `explode` and group the result:

```python
sliding = (
    clicks.select(w=bt.window(col("ts"), "1h", "30m"), n=col("n"))
    .explode("w")
    .group_by("w")
    .agg(hits=col("n").sum())
    .sort("w")
)
print(sliding.to_pydict()["hits"])
# [1, 3, 2, 3, 3]
```

Five windows, from 23:30 the previous day through 01:30, and each click is counted in both
windows that contain it. The 00:00 window holds the 00:00 and 00:30 clicks, so it sums to 3.

:::{warning}
Grouping by a sliding window directly, {py:meth}`group_by(w=bt.window(col("ts"), "1h", "30m")) <batcher.Dataset.group_by>`,
would group by the *list* rather than by the windows, counting each row once instead of
once per window it belongs to. That is a wrong answer, so the engine rejects it and points
at `explode`. A tumbling window (no slide) is a single start and groups directly.
:::

:::{important}
Watermark-driven window eviction recognizes tumbling windows only. After the `explode` a
sliding aggregation groups by an ordinary column, so the engine cannot tell which window is
closed, and it will not evict one. On an unbounded source that means the aggregation's state
grows without bound. Sliding windows are a batch operation today; use a tumbling window for
a long-running stream.
:::

On an unbounded stream, declare a **watermark** so windowed state stays bounded:
{py:meth}`ds.with_watermark(time_col, lateness) <batcher.Dataset.with_watermark>` lets the engine emit and evict a window once
the watermark (`max(event_time) - lateness`) passes its end, and drop rows that
arrive later than that. The query is otherwise identical to the batch one:

```python
# docs: skip
windowed = (
    bt.read.kafka(topic="clicks")
    .with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1h"))
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

{py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` keeps the first row per key seen inside the
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

## Stream-to-stream joins

{py:meth}`join_stream <batcher.Dataset.join_stream>` joins two streams on keys **and** an event-time interval
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

`how=` takes `"inner"` (the default), `"left"`, `"right"`, and `"full"`. An unmatched row
is emitted null-padded at the moment the watermark guarantees no partner can still arrive
for it — which is the only moment that statement is decidable about an unbounded stream,
and why the interval is required rather than optional:

```python
unclicked = impressions.join_stream(
    clicks2, on="ad", left_time="shown", right_time="clicked", within="5m", how="left"
)
print(sorted(unclicked.to_pydict()["ad"]))  # ['a', 'b'] — 'b' with null click columns
```

A joined stream writes to a sink like any other streaming query:

```python
# docs: skip
attributed.write.delta("lake/attribution", trigger=bt.Trigger.processing_time("30 seconds"))
```

`checkpoint=` is the one thing it refuses. A join's state is two buffered sides and two
watermarks, none of it addressable by a source offset, so there is nothing to resume from
— and accepting the argument would restart from an empty join on every restart while
looking exactly like exactly-once recovery. The sink's own idempotency still applies, so a
replayed micro-batch does not duplicate rows.

## Exactly-once and checkpointing

Pass `checkpoint=<dir>` to a streaming write to record source offsets and sink
commits per micro-batch. On restart the query resumes from the last committed offset.
A replayable source (Kafka offsets, Kinesis sequence numbers, a Delta version, the
Auto-Loader seen-file set) seeks forward and an idempotent sink dedups, so the
combined output is exactly-once, with no row lost or duplicated.

The sink's half of that is worth being concrete about, because replay is not optional:
the engine records a micro-batch's source offset *before* it processes the batch, so a
crash in between leaves a batch the next run **will** re-emit. A plain append would then
write those rows twice. Writing to Delta, each micro-batch instead commits with a
transaction id, the query name plus the batch number, and the sink checks the log for
it first. A replayed batch finds its own transaction already recorded, writes no file and
commits nothing. That is what turns the engine's at-least-once replay into end-to-end
exactly-once, and it is why the log holds exactly one transaction per micro-batch however
many times one was retried.

Give the query a stable `query_name` if you rely on this: the name is the transaction's
application id, so it has to be the same across restarts for the check to find the
previous run's commits. Without one it is derived from the destination table, which is
stable but shared, so two different unnamed queries writing the same table would collide.

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

## Running the stream on a cluster

Add `distributed=True` and each micro-batch runs as one **epoch across the cluster**
instead of on the driver. The workers read their share of the epoch, run the pipeline,
and write their own data files; the driver never touches a row.

What it does *not* do is commit once per worker. The workers write their files without
committing them, and the driver then publishes the whole epoch as a **single**
transaction. So the guarantees above survive the fan-out unchanged:

- **one transaction per micro-batch**, whatever the worker count. The log still reads as
  a record of the stream, not of the machines that ran it.
- **exactly-once**, because that one commit carries the micro-batch's transaction id. A
  replayed epoch, from a lost worker or a restart, finds itself already committed and
  writes nothing.

The source's offsets are written to the checkpoint *between* staging an epoch and
publishing it. That bounds a crash to an epoch that was staged and never published, which
is one the next run safely replays.

```python
# docs: skip
# New files land continuously; each arrival becomes one micro-batch, fanned across
# the cluster, and one Delta transaction.
q = (bt.read.files_incremental("lake/landing", "parquet", state_dir="lake/bronze/_seen")
       .filter(col("status") == "ok")
       .write.delta("lake/bronze",
                    trigger=bt.Trigger.processing_time("1 minute"),
                    checkpoint="lake/bronze/_ck",
                    query_name="bronze-ingest",
                    distributed=True, num_workers=16))
q.stop()   # the query runs until you stop it — an idle minute is not the end of a stream
```

A streaming aggregation distributes too. Each worker aggregates only its share of the
epoch and returns a partial result, which the driver merges. That is the same
`partial`, `combine`, `finalize` sequence the single-node aggregate uses, so the answer is
identical.

Write to Delta if you want the exactly-once guarantee. Iceberg has no transaction-id
check, so a replayed micro-batch there would duplicate rows rather than be recognized as
already-committed. A distributed streaming write to it is refused rather than quietly
giving you a weaker guarantee than this page promises.

## The medallion pattern

Each layer reads the previous one as a stream, so the three medallion layers chain
with the primitives above: an incremental file or Delta read in, a transform, a
checkpointed write out.

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

- {doc}`/user-guide/moving-data/writing-data`: the batch write surface `ds.write` extends.
- {doc}`/user-guide/analyze/aggregations` and {doc}`/user-guide/analyze/window-functions`: the grouping and SQL-window APIs.
- {doc}`/architecture/execution`: the pipelines-and-breakers execution model that
  makes batch and streaming one engine.
- {doc}`/ml/inference/streaming`: streaming a query as bounded-memory training data.
- {doc}`/agents`: the `write-a-streaming-pipeline` agent skill covers this
  surface as a procedure, including the batch-vs-stream return-type trap.
- {doc}`/cookbook/operations/streaming_basics`: the same operators run incrementally, as a script.
