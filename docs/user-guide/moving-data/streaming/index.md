# Streaming

This page covers streaming in Batcher: reading unbounded sources, writing them with a trigger, event-time windows and watermarks, and exactly-once restart. Batcher treats batch as the bounded special case of streaming, so one {py:class}`Dataset <batcher.Dataset>` API (`group_by`, `window`, `join`, `with_columns`, `write`) runs over a finite table and an unbounded stream alike. Turning a one-off job into a continuous one means changing the *source*, or adding a `trigger`. There is no second API to learn.

## One API, batch or streaming

Every operation on this page works on a bounded dataset, which returns a result you can {py:meth}`collect() <batcher.Dataset.collect>`, and on an unbounded one, which you consume with {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` or write to a sink. The query is identical:

```python
import batcher as bt
from batcher import col

events = bt.from_pydict({"user": ["a", "b", "a", "c"], "amount": [10, 5, 7, 3]})

# Bounded source → a finite result.
totals = events.group_by("user").agg(total=col("amount").sum())
print(totals.to_pydict())
```

Point the same transformation at an unbounded source and consume it incrementally:

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

An unbounded source can't `collect()`, since it would never finish. A terminal that tries to materialize one raises a {py:exc}`PlanError <batcher.PlanError>`, and {py:obj}`ds.is_streaming <batcher.Dataset.is_streaming>` tells you which kind you hold.

Not every shape emits while the stream runs. An unwatermarked `group_by(...).agg(...)`, an uncapped `distinct()` and a top-N fold their input and emit once, at end of input. {doc}`/user-guide/moving-data/streaming/emission` has the whole list.

## Reading streams

Streaming sources sit behind the same {py:obj}`bt.read <batcher.read>` namespace as files and tables. The following table lists them:

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

`files_incremental` forwards any option it doesn't recognize to the file reader for each new file. A watched directory is configured the way a batch read of it would be: a CSV `delimiter`, a declared `schema`, `storage_options` for the store, and the two tolerance flags. Those flags matter more on a stream than in a batch job. The files come from a producer nobody is watching, and a query that can't tolerate one bad record stops on it and stays stopped.

```python
# docs: skip
q = bt.read.files_incremental(
    "s3://bucket/landing/",
    "csv",
    state_dir="s3://bucket/_seen",
    delimiter=";",
    on_error="skip",  # a truncated upload drops the file
    on_bad_lines="skip",  # a ragged line drops the line
).write.delta("s3://lake/bronze", trigger="30 seconds")
```

An option the reader doesn't accept is refused when the query is built, not when the file carrying it arrives.

Kinesis, Pulsar and Event Hubs share one `starting_position=` option, `"earliest"` or `"latest"`, and map it onto their own vocabulary (Kinesis's `ShardIteratorType`, Pulsar's `InitialPosition`, Event Hubs' offset sentinel). Each connector's native spelling still works. Kafka takes the Spark spelling, `starting_offsets=`, which accepts the same two values or a `{partition: offset}` map.

The `rate` source generates `(value, timestamp)` rows, so you can try the API without external infrastructure. `num_rows` bounds it and `pace=False` removes the one-second cadence:

```python
demo = bt.read.rate(5, num_rows=10, pace=False)
rows = [b.num_rows for b in demo.iter_batches()]
print(sum(rows))  # 10 generated (value, timestamp) rows
```

`rate` promises rows per *second*, so how many land in a micro-batch depends on how long the previous one took. That makes it a poor benchmark input: the thing being measured changes the input. `rate_micro_batch` promises rows per *batch* instead, so a run is reproducible:

```python
bench = bt.read.rate_micro_batch(4, num_rows=8)
print([b.num_rows for b in bench.iter_batches()])
# [4, 4]
```

The broker readers need their optional dependency and a running service:

```python
# docs: skip
clicks = bt.read.kafka(topic="clicks", bootstrap_servers="localhost:9092")
recent = clicks.filter(col("partition") == 0)
for batch in recent.iter_batches():
    handle(batch)
```

### Look at a stream before you build on it

A materializing terminal on an unbounded source has no finite answer, so `to_pydict()` and its relatives refuse it. A query capped by `limit(n)`, and {py:meth}`show <batcher.Dataset.show>`, are the exception, because their answer *is* finite. The engine stops reading the moment it has the rows, which makes this the right first thing to type against a topic that never ends:

```python
peek = bt.read.rate(5, num_rows=100, pace=False).limit(3)
print(peek.count())
# 3
```

`distinct()` before the limit is finite for the same reason. Once the engine has seen `n` distinct rows it stops, because every later row is either a duplicate or arrives too late to displace one of the first `n`:

```python
values = bt.read.rate(5, num_rows=100, pace=False).select("value").distinct().limit(3)
print(values.to_pydict())
# {'value': [0, 1, 2]}
```

A limit over a *sort* still refuses. Top-N is finite too, but not knowable until the last row has arrived. A keyed `distinct(subset=...)` refuses as well, because which row survives per key depends on rows that haven't arrived. Use {py:meth}`drop_duplicates_within_watermark <batcher.Dataset.drop_duplicates_within_watermark>` when you need one row per key off a stream.

## Write a stream with `ds.write`

{py:obj}`ds.write(...) <batcher.Dataset.write>` is the one write surface. With a bounded source and no trigger it is a single batch write that returns a {py:class}`WriteManifest <batcher.io.WriteManifest>`. Add a `trigger=`, or point it at an unbounded source, and it runs as a streaming query that appends each micro-batch and hands back a `StreamingQuery`:

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

The following table lists the streaming sinks on the write namespace:

| Sink | What each micro-batch does |
| --- | --- |
| `ds.write(path, format=..., trigger=...)` | Writes Parquet, CSV or JSON as `part-batch*` files, idempotent on restart. |
| {py:meth}`ds.write.delta(uri, trigger=...) <batcher.api.io_namespace.writer.Writer.delta>` | Commits one transactional Delta append. |
| `ds.write(table, "dbapi", uri=..., mode="upsert", key_columns=..., trigger=...)` | Commits one database transaction. |
| {py:meth}`ds.write.memory(name, trigger=...) <batcher.api.io_namespace.writer.Writer.memory>` | Appends to an in-memory table you read with {py:func}`bt.read_memory(name) <batcher.read_memory>`. |
| {py:meth}`ds.write.console(trigger=..., num_rows=..., truncate=...) <batcher.api.io_namespace.writer.Writer.console>` | Prints the batch. Development only. |
| {py:meth}`ds.write.for_each_batch(fn, trigger=...) <batcher.api.io_namespace.writer.Writer.for_each_batch>` | Calls `fn(table, batch_id)` with the whole Arrow table. |
| {py:meth}`ds.write.for_each(fn, trigger=...) <batcher.api.io_namespace.writer.Writer.for_each>` | Calls `fn(row)` per row. |
| {py:meth}`ds.write.kafka(topic, bootstrap_servers=..., trigger=...) <batcher.api.io_namespace.writer.Writer.kafka>` | Publishes each row to Kafka. |
| {py:meth}`ds.write.noop(trigger=...) <batcher.api.io_namespace.writer.Writer.noop>` | Runs the pipeline and discards the output. |

A few of them carry a contract worth knowing. The `dbapi` upsert makes a replayed batch a no-op, so a database table is exactly-once with no transaction log, and the operational stores (`mongo`, `dynamodb`, `cassandra`, `redis`, `elasticsearch`, `hbase`) work the same way. See {doc}`Writing to a database </integrations/databases/writing>`. `for_each_batch` never hands you a row, which makes it the hook for a multi-statement commit, an SCD, or a multi-sink fan-out.

`for_each` accepts a {py:class}`ForeachWriter <batcher.ForeachWriter>` in place of a function when the destination needs a connection. Its `open(partition_id, epoch_id)` acquires one and returns whether to proceed, `process(row)` writes a row, and `close(error)` releases it, including when the epoch failed. A bare function has nowhere to put a connection.

`write.kafka` follows Spark's column contract: `value` is required, and `key`, `topic`, `partition` and `headers` are optional. Delivery is at-least-once, so make the consumer idempotent or dedup on the key. The console sink shortens strings to 20 characters unless you pass `truncate=False` or an explicit width. `noop` is the benchmark sink, because measuring through a real sink measures the sink too. It still counts rows, so `recent_progress` reports what the query processed.

### Triggers

A {py:class}`Trigger <batcher.Trigger>` sets the cadence, with Spark's names. The following table compares them:

| Trigger | Behavior |
| --- | --- |
| {py:meth}`bt.Trigger.processing_time("5 seconds") <batcher.Trigger.processing_time>` | Fires a micro-batch on a wall-clock interval. The default streaming cadence. |
| {py:meth}`bt.Trigger.available_now() <batcher.Trigger.available_now>` | Drains every record available at start, then stops. The incremental-batch and backfill trigger. |
| {py:meth}`bt.Trigger.once() <batcher.Trigger.once>` | The same execution as `available_now()`. |
| {py:meth}`bt.Trigger.continuous("1 second") <batcher.Trigger.continuous>` | Runs micro-batches back to back and commits a checkpoint epoch on the interval. Stateless pipelines only. |

Spark's `Once` forces everything into a single micro-batch and was deprecated for that reason. Batcher's drains across as many micro-batches as the data needs, so prefer `available_now()` in new code: same behavior, the name Spark now recommends.

### Output modes

`output_mode=` controls what each micro-batch emits:

- `"append"` (default) emits only rows that won't change again. For a plain pipeline that is every row. For a windowed aggregation it is a window's row once the watermark closes it.
- `"complete"` emits the full result table after every micro-batch. Aggregations only, including the projections and HAVING filters above one ({doc}`emission`).
- `"update"` emits only the result rows whose value changed this micro-batch.

Those literals are the values of the {py:class}`OutputMode <batcher.OutputMode>` constants {py:obj}`bt.OutputMode.APPEND <batcher.OutputMode.APPEND>`, {py:obj}`bt.OutputMode.COMPLETE <batcher.OutputMode.COMPLETE>` and {py:obj}`bt.OutputMode.UPDATE <batcher.OutputMode.UPDATE>`. Pass a constant for a typo-proof spelling.

```python
print(bt.OutputMode.COMPLETE)
# complete
```

```python
agg_stream = (
    bt.from_batches(feed, schema, bounded=False).group_by("user").agg(total=col("amount").sum())
)
q = agg_stream.write.memory(
    "running_totals", trigger=bt.Trigger.available_now(), output_mode="complete"
)
q.await_termination()
print(
    dict(
        zip(
            *[bt.read_memory("running_totals").to_pydict()[c] for c in ("user", "total")],
            strict=True,
        )
    )
)
```

### Size the files a stream leaves behind

A file-sink stream writes one file per micro-batch, so file size is whatever the trigger interval happened to produce. Over days that is the small-files problem in its purest form. `max_rows_per_file` caps each output file and splits a micro-batch across as many files as it needs. The chunk index joins the batch id in the name, so a replayed epoch still recognizes its own output.

```python
import glob
import os
import tempfile

capped_dir = os.path.join(tempfile.mkdtemp(), "capped")
q = bt.from_pydict({"v": list(range(20))}).write(
    capped_dir, format="parquet", trigger=bt.Trigger.available_now(), max_rows_per_file=6
)
q.await_termination()
print(len(glob.glob(os.path.join(capped_dir, "*.parquet"))))
# 4
```

The cap applies to file sinks. A transactional target (Delta, Iceberg, Hudi) makes each micro-batch one transaction and owns the file layout inside it, so the option is refused there with a pointer to {py:obj}`bt.compact <batcher.compact>`. A `distributed=True` stream refuses it too, because its files are named for their epoch and shard and aren't subdivided further.

## Monitor a running query

The `StreamingQuery` a streaming write returns carries the query's liveness, per-micro-batch metrics, the state each stateful operator holds, and the count of rows dropped as late. {doc}`monitoring` covers all of it.

## Event-time windows and watermarks

`bt.window(time_col, duration)` assigns each row to one event-time window. Group by it like any other key, batch or streaming:

```python
import datetime as dt

base = dt.datetime(2024, 1, 1)
clicks = bt.from_pydict(
    {
        "ts": [base, base + dt.timedelta(minutes=30), base + dt.timedelta(minutes=90)],
        "n": [1, 2, 3],
    }
)
hourly = clicks.group_by(w=bt.window(col("ts"), "1h")).agg(hits=col("n").sum())
print(hourly.to_pydict())  # 00:00 → 3, 01:00 → 3
```

### Sliding windows explode, they don't group

Pass a third argument and the windows overlap. `bt.window(col("ts"), "1h", "30m")` is a one-hour window advancing every thirty minutes, so one row belongs to *two* windows, and the expression evaluates to the **list** of window starts containing the row. Fan that list out with `explode` and group the result:

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

That's five windows, from 23:30 the previous day through 01:30, with each click counted in both windows that contain it. The 00:00 window holds the 00:00 and 00:30 clicks, so it sums to 3.

:::{warning}
Grouping by a sliding window directly, {py:meth}`group_by(w=bt.window(col("ts"), "1h", "30m")) <batcher.Dataset.group_by>`, would group by the *list* and count each row once instead of once per window. That is a wrong answer, so the engine rejects it and points at `explode`. A tumbling window (no slide) is a single start and groups directly.
:::

:::{note}
Watermark-driven eviction recognizes the sliding shape too. After the `explode` the group key is an ordinary column, so the engine reads the width and hop from the `window(..., slide)` beneath it and closes each window when the watermark passes its end. State is bounded by the number of *open* windows, which for overlapping windows is `width / slide` rather than one.
:::

On an unbounded stream, declare a *watermark* so windowed state stays bounded. {py:meth}`ds.with_watermark(time_col, lateness) <batcher.Dataset.with_watermark>` lets the engine emit and evict a window once the watermark, `max(event_time) - lateness`, passes its end, and drop rows that arrive later than that. The query is otherwise the batch one:

```python
# docs: skip
windowed = (
    bt.read.kafka(topic="clicks")
    .with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1h"))
    .agg(hits=col("n").sum())
)
windowed.write.delta(
    "gold/hourly",
    trigger=bt.Trigger.processing_time("1 minute"),
    output_mode="append",
    checkpoint="gold/_ckpt",
)
```

*Session windows* group consecutive events whose gap is below a timeout:

```python
sessions = clicks.session_window("ts", "45m", hits=col("n").sum())
print(sessions.select("session_start", "session_end", "hits").to_pydict())
```

## Operators that remember

Deduplication within a watermark, the stream-stream interval join, the session window, arbitrary keyed state and the union of two streams all keep something between micro-batches. Each is bounded by a limit you choose rather than by the data. {doc}`stateful` covers them.

Joining a stream to a static table needs none of that. Write it as an ordinary {py:meth}`join <batcher.Dataset.join>`: the table is read once when the query starts, and every micro-batch joins against all of it. Join types that would need the *static* side to be complete are refused, which is where Spark draws the line too. {doc}`/cookbook/streaming/stream-join` walks through that join end to end, and {doc}`/cookbook/streaming/windowed-aggregation` does the same for the session window.

## Exactly-once and checkpointing

Pass `checkpoint=<dir>` to a streaming write to record source offsets and sink commits per micro-batch, and a restart resumes from the last committed offset. A replayable source (Kafka offsets, Kinesis sequence numbers, a Delta version, the Auto Loader seen-file set) seeks forward and an idempotent sink dedups. Together they give exactly-once output, with no row lost or duplicated.

Replay is not optional, so the sink's half matters. The engine records a micro-batch's source offset *before* it processes the batch, and a crash in between leaves a batch the next run **will** re-emit. A plain append would write those rows twice. The Delta sink commits each micro-batch with a transaction id, the query name plus the batch number, and checks the log for it first. A replayed batch finds its own transaction already recorded, writes no file and commits nothing. That turns at-least-once replay into end-to-end exactly-once, and it's why the log holds exactly one transaction per micro-batch however many times one was retried.

The order inside one micro-batch is what makes replay safe, and it's the same under every trigger:

![One micro-batch as a cycle. A trigger fires on a processing-time interval or as an available_now drain. The engine stages the epoch, reading and computing while publishing nothing; writes the source position it consumed ahead of publishing anything; hands the rows to the sink; then snapshots state and commits with the sink's token. A dashed edge sleeps the rest of the interval before the next trigger, and a draining trigger skips that wait and stops when the source is spent. Because the position is durable before anything is published, the only epoch a crash can lose is one that was staged and not published, and the next run replays it into a sink that records its own query name and batch id and so commits nothing the second time.](/_static/diagrams/streaming_microbatch.svg)

Give the query a stable `query_name` if you rely on this. The name is the transaction's application id, so it must stay the same across restarts for the check to find the previous run's commits. Without one it's derived from the destination table, which is stable but shared: two unnamed queries writing the same table would collide.

```python
# docs: skip
q = bt.read.kafka(topic="orders").write(
    "lake/bronze",
    format="parquet",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="lake/bronze/_checkpoint",
)
# A crash and restart of the same query against the same checkpoint resumes
# exactly where it left off.
```

## Run the stream on a cluster

Add `distributed=True` and each micro-batch runs as one epoch across the cluster instead of on the driver. The workers read their share of the epoch, run the pipeline and write their own data files. The driver never touches a row.

Workers don't commit. They write their files uncommitted, and the driver publishes the whole epoch as a single transaction, so the guarantees above survive the fan-out unchanged. The log holds one transaction per micro-batch whatever the worker count, and that commit carries the micro-batch's transaction id, so a replayed epoch from a lost worker or a restart finds itself committed and writes nothing. The source offsets are checkpointed *between* staging an epoch and publishing it, which bounds a crash to an epoch that was staged and never published.

```python
# docs: skip
# New files land continuously; each arrival becomes one micro-batch, fanned across
# the cluster, and one Delta transaction.
q = (
    bt.read.files_incremental("lake/landing", "parquet", state_dir="lake/bronze/_seen")
    .filter(col("status") == "ok")
    .write.delta(
        "lake/bronze",
        trigger=bt.Trigger.processing_time("1 minute"),
        checkpoint="lake/bronze/_ck",
        query_name="bronze-ingest",
        distributed=True,
        num_workers=16,
    )
)
q.stop()  # the query runs until you stop it; an idle minute is not the end of a stream
```

A streaming aggregation distributes too. Each worker aggregates its share of the epoch into a partial result and the driver merges them, the same `partial`, `combine`, `finalize` sequence the single-node aggregate uses, so the answer is identical.

Write to Delta for a distributed exactly-once stream. Iceberg's writer has no transaction-id check, so a replayed micro-batch would duplicate rows, and a distributed streaming write to it is refused rather than quietly giving you a weaker guarantee.

## The medallion pattern

Each medallion layer reads the previous one as a stream, so the three chain with the primitives above: an incremental read in, a transform, a checkpointed write out.

```python
# docs: skip
# Bronze: raw ingestion.
bt.read.kafka(topic="events").write(
    "lake/bronze",
    format="parquet",
    trigger=bt.Trigger.available_now(),
    checkpoint="lake/bronze/_ck",
)

# Silver: clean + dedup, reading bronze incrementally.
(
    bt.read.files_incremental("lake/bronze", "parquet", state_dir="lake/silver/_seen")
    .drop_duplicates_within_watermark(["id"], event_time="ts", lateness="10m")
    .write(
        "lake/silver",
        format="parquet",
        trigger=bt.Trigger.available_now(),
        checkpoint="lake/silver/_ck",
    )
)

# Gold: windowed aggregates, reading silver incrementally.
(
    bt.read.files_incremental("lake/silver", "parquet", state_dir="lake/gold/_seen")
    .with_watermark("ts", "10m")
    .group_by(w=bt.window(col("ts"), "1h"))
    .agg(total=col("v").sum())
    .write.delta(
        "lake/gold",
        trigger=bt.Trigger.available_now(),
        output_mode="append",
        checkpoint="lake/gold/_ck",
    )
)
```

## See also

- {doc}`emission`: which shapes emit while a stream runs.
- {doc}`stateful`: dedup, interval joins, keyed state and union.
- {doc}`monitoring`: progress, state size and late rows.
- {doc}`/user-guide/moving-data/writing-data`: the batch write surface `ds.write` extends.
- {doc}`/user-guide/analyze/aggregations` and {doc}`/user-guide/analyze/window-functions`: the grouping and SQL-window APIs.
- {doc}`/architecture/execution`: the pipelines-and-breakers execution model that makes batch and streaming one engine.
- {doc}`/ml/inference/streaming`: streaming a query as bounded-memory training data.
- {doc}`/agents`: the `write-a-streaming-pipeline` agent skill covers this surface as a procedure.
- {doc}`/cookbook/operations/streaming_basics`: the same operators run incrementally, as a script.

```{toctree}
:hidden:

emission
stateful
monitoring
```
