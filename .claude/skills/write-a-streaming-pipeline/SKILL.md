---
name: write-a-streaming-pipeline
description: Write, run, and monitor a continuous Batcher streaming query — unbounded sources, triggers and output modes, watermarks and late data, streaming joins and windowed aggregations, checkpointing and exactly-once restart semantics, streaming sinks, and graceful shutdown. Invoke when a pipeline reads an unbounded source (Kafka/Kinesis/Pulsar/PubSub/EventHubs/socket/rate/files_incremental), uses a trigger or checkpoint, or must run continuously rather than once.
---

# Write a streaming pipeline

Streaming reuses the whole relational surface — `filter`, `join`, `group_by`, expressions
all mean the same thing. Read `write-a-batcher-pipeline` for those; this skill covers only
what unboundedness adds. Batch is the bounded special case of streaming, so a stream is
**the same plan with a different source and a different terminal**.

## The two traps — read these first

Both were verified against the source, and both bite agents immediately.

### Trap 1: the same `write` call returns two different types

`writer.py` switches on **"a trigger is set OR any source is unbounded"**, so one call site
returns a `WriteManifest` (batch) or a live `StreamingQuery` (stream) depending on the
*source*, not on how the call is written:

```python
bt.from_pydict({"x": [1, 2, 3]}).write.parquet(path)        # -> WriteManifest
bt.read.rate(rows_per_second=1000).write.parquet(path)      # -> StreamingQuery
```

`write.parquet` is even **annotated `-> WriteManifest`** while returning a
`StreamingQuery` — its `**opts` forward to `Writer.__call__`, so the hint lies, and **an
unbounded source needs no trigger to trip it.** Treating a live query handle as a finished
manifest means the program exits mid-query, or reads `manifest.paths` off an object that
has none. Check `ds.is_streaming` before writing, and always `await_termination`/`stop`
after a streaming write.

### Trap 2: `output_mode="complete"` to a path sink raises

```python
bt.read.rate(...).write.parquet(path, output_mode="complete")
# PlanError: streaming write to a path sink (...) supports output_mode='append' only
```

A file/Delta sink **appends** each micro-batch, so a running `complete`/`update` aggregate
would land as another part file every trigger and readback would silently duplicate the
result. Batcher refuses rather than produce wrong data. For a running aggregate use
`.write.memory(name, output_mode="complete")` or `.write.for_each_batch(fn)` with your own
upsert. (The error text suggests `.write.foreach_batch` — that method does not exist; the
public name is **`for_each_batch`**.)

## Sources

```python
bt.read.kafka("topic", bootstrap_servers="...", group="batcher")
bt.read.kinesis("stream", region="us-east-1")
bt.read.pulsar("topic", service_url="pulsar://...", subscription="batcher")
bt.read.pubsub("topic");  bt.read.eventhubs("topic", connection_str="...")
bt.read.socket("localhost", 9999)                    # dev
bt.read.rate(rows_per_second=5, num_rows=20)         # dev/test generator
bt.read.files_incremental("s3://drop/", "parquet", state_dir="s3://state/")  # autoloader
```

`bt.read.rate` is the one you can actually run in a test: timestamps are deterministic
(spaced from the Unix epoch, not wall-clock), so results reproduce. `num_rows` bounds the
generator and already disables pacing. `files_incremental` is the Databricks `cloudFiles`
analog — it tracks processed files in a SQLite `SeenStore` under `state_dir`, so a re-run
never reprocesses one; files stay *pending* until their epoch is published, so a crash
re-finds them and the sink's per-batch transaction makes the replay idempotent.

`ds.is_streaming` tells you whether a plan is unbounded. An unbounded plan **cannot be
`collect()`ed** — use `iter_batches()` or a streaming write.

## Triggers — when a micro-batch fires

```python
bt.Trigger.available_now()          # drain all available data, then stop  <- use in tests
bt.Trigger.once()                   # exactly one micro-batch, then stop
bt.Trigger.processing_time("5s")    # a micro-batch every 5s   <- the production default
bt.Trigger.continuous("1s")         # low-latency, STATELESS pipelines only
```

Every trigger exposes `.kind` and `.interval_seconds`; build them with these classmethods,
never the raw constructor. `continuous` supports only stateless work (filter/select/
`map_batches`) — no aggregation or join, as in Spark. Choose `available_now` for a backfill
or a test that must terminate; `processing_time` for a live pipeline (the interval is your
latency/efficiency dial); `once` for an externally-scheduled tick.

**Two different duration parsers — a genuine footgun.** Trigger intervals accept both
`"5s"` and `"5 seconds"`; watermarks, windows, and `within` accept **only the compact
offset form** (`y/mo/w/d/h/m/s`, e.g. `"10s"`, `"1mo15d"`):

```python
bt.Trigger.processing_time("5 seconds")   # fine
ds.with_watermark("ts", "5 seconds")      # ValueError: invalid offset '5 seconds'
ds.with_watermark("ts", "5s")             # correct
```

## Output modes

`bt.OutputMode.APPEND` / `.COMPLETE` / `.UPDATE` are plain `str` constants (not an Enum);
`bt.OutputMode.validate(mode)` returns the string or raises.

- **append** — only rows final and never changing. The only mode a path sink accepts; needs
  a watermark when the query aggregates, so a window can be known closed.
- **complete** — whole result table re-emitted each micro-batch; aggregations only, must fit
  the sink. **update** — only changed rows; the sink upserts by key.

## Watermarks and late data

A watermark is the promise "no more events older than this". It advances to
`max(observed event time) - lateness`; older rows are dropped as late, and windows whose
end falls below it are emitted and **evicted** — which is what keeps state bounded.

```python
events = (
    bt.read.kafka("clicks")
    .with_watermark("ts", "10m")                       # tolerate 10m of lateness
    .group_by(w=bt.window(bt.col("ts"), "1h"))         # tumbling; add a 3rd arg to slide
    .agg(hits=bt.count())
)
```

`lateness` is the whole trade: larger means fewer dropped late events but more retained
state and later emission. Set it from your source's measured lag, not by taste.

```python
ds.session_window("ts", "45m", partition_by=["user"], hits=bt.count())      # gap-based
ds.drop_duplicates_within_watermark(["event_id"], event_time="ts", lateness="10m")
```

`session_window` groups events whose inter-arrival gap is under `gap`, emitting
`session_start`/`session_end` plus the named aggregates.
`drop_duplicates_within_watermark` keeps the first row per key within the watermark — the
standard defense against an at-least-once upstream.

## Streaming joins

```python
impressions.join_stream(
    clicks, on="ad_id",
    left_time="imp_ts", right_time="click_ts",
    within="30m", lateness="10m",
)
```

A watermark-bounded **interval inner join** with **no `how=` parameter** — inner is the only
option. A pair matches when `|left_time - right_time| <= within`, and that bound is what
lets buffered state be evicted. If every source is bounded it degrades to a plain inner join
plus an interval filter, so the same code tests offline. Joining a stream to a small
**bounded** dimension table is an ordinary `.join(...)`.

## Checkpointing, restart, and exactly-once

Pass a `checkpoint=` directory and a **stable `query_name=`** to any streaming write:

```python
q = bt.read.kafka("orders").write(
    "s3://lake/bronze/orders/", format="parquet",
    trigger=bt.Trigger.processing_time("30s"),
    checkpoint="s3://state/orders/", query_name="orders-ingest",
)
```

The checkpoint dir holds `offsets.sqlite`, `commits.sqlite`, and a `state/` dir of Arrow IPC
snapshots. Per micro-batch the order is **offsets write-ahead → state snapshotted → sink
commits → commit-log entry last**, so a batch in offsets but not in commits is exactly the
in-flight batch, and restart replays that one only. Verified: draining the same source
twice against one checkpoint left the output at 20 rows, not 40.

**The engine is deliberately at-least-once; end-to-end exactly-once is bought by the
sink.** Know which one you have:

- **Delta** — commits each micro-batch with a transaction id `(app_id, batch_id)` and checks
  the log first, so a replay writes nothing. Exactly-once.
- **File/path** — one atomic `part-batch<NNNNN>` per micro-batch, idempotent *by position*;
  exactly-once only if the plan is deterministic.
- **Iceberg** — **no** transaction-id check; a replayed micro-batch duplicates rows.
- **`for_each_batch`** — no idempotency at all; you get `batch_id`, build your own key.

`query_name` is the transaction application id and **must be stable across restarts**, or
the idempotency check never finds the previous run's transactions. Unnamed queries derive
it from the destination, so two unnamed queries on one table collide — always name them.
Changing the checkpoint location resets the stream to the beginning. A `files_incremental`
source keeps its own `state_dir` alongside the checkpoint; both must survive a restart.

## Sinks

```python
ds.write(path, format="parquet", trigger=..., checkpoint=..., query_name=...)  # path/lakehouse
ds.write.console(num_rows=20)                      # debugging
ds.write.memory("name", output_mode="complete")    # read back with bt.read_memory("name")
ds.write.for_each_batch(fn)                        # fn(table: pa.Table, batch_id: int)
ds.write.for_each(fn)                              # fn(row: dict) — convenience, slower
```

`bt.read.table(name)` raises `unknown source` — a memory sink reads back only via
**`bt.read_memory(name)`**; in `complete` mode it replaces its table each batch, else grows.
`for_each_batch` is the sanctioned hook for custom upserts (MERGE/SCD), multi-sink fan-out,
and per-batch quality gates: it receives a whole Arrow table, never a row.

## Running, monitoring, and shutting down

A streaming write returns a `StreamingQuery`. Note the property/method split:

```python
q.name / q.is_active / q.status / q.last_progress   # PROPERTIES
q.recent_progress()                 # METHOD -> list[StreamingQueryProgress]
q.exception()                       # METHOD -> BaseException | None (does not raise)
q.await_termination(timeout=None)   # -> bool; RE-RAISES a query failure
q.stop()
```

`status` is `StreamingQueryStatus(is_active, is_data_available, is_trigger_active, message,
batches_processed)`; `last_progress` is `StreamingQueryProgress(batch_id, num_input_rows,
num_output_rows, duration_ms, timestamp)` plus `input_rows_per_second` — dataclasses, not
dicts. `bt.streams()` lists active queries; `bt.await_any_termination(timeout=None)` blocks
until one stops. Both `stop()` and a completed `await_termination()` deregister the query,
so `bt.streams()` is `[]` afterward.

**`StreamingQuery` is a context manager — that is the graceful-shutdown idiom**, because
`__exit__` calls `stop()` even if the body raised:

```python
with bt.read.kafka("orders").write.console(trigger=bt.Trigger.processing_time("5s")) as q:
    while q.is_active:
        if p := q.last_progress:
            print(p.batch_id, p.num_input_rows, p.input_rows_per_second)
        q.await_termination(timeout=10)
```

### A complete example that runs locally (no Kafka)

```python
stream = bt.read.rate(rows_per_second=5, num_rows=20, pace=False)
q = stream.filter(bt.col("value") % 2 == 0).write.memory(
    "evens", trigger=bt.Trigger.available_now(), query_name="evens"
)
q.await_termination(timeout=30)
print(q.status, q.last_progress, bt.read_memory("evens").count())
```

Real output — 20 rows drained as 4 micro-batches of 5, 10 even values kept:

```
StreamingQueryStatus(is_active=False, ..., message='Stopped', batches_processed=4)
StreamingQueryProgress(batch_id=3, num_input_rows=5, num_output_rows=2, ...) 10
```

## Self-check

- [ ] You know whether the write returned a `WriteManifest` or a `StreamingQuery` — checked
      `ds.is_streaming`, not the (lying) type annotation.
- [ ] Every streaming write is followed by `await_termination`/`stop`, or wrapped in `with`;
      the program cannot exit with a query still live.
- [ ] `output_mode` is `append` for any path/Delta sink; `complete`/`update` go to memory or
      `for_each_batch`. Durations use the compact form (`"10m"`), not `"10 minutes"`.
- [ ] Any streaming aggregation has a `with_watermark`, so state stays bounded.
- [ ] `checkpoint=` **and** a stable `query_name=` are set on every production write, and the
      exactly-once story is explicit (Delta/file sink, or `for_each_batch` keyed on
      `batch_id`) — not assumed.
- [ ] No `collect()` on an unbounded plan; `continuous` only on stateless pipelines.
- [ ] `recent_progress()`/`exception()` called as methods; `status`/`last_progress` read as
      properties. Tests use `bt.read.rate(num_rows=N)` + `available_now()` so they end.

## See also

- `docs/user-guide/moving-data/streaming.md` (exactly-once at :297, cluster at :331, medallion at :374);
  `docs/examples/streaming/{kafka-etl,exactly-once-sink,late-data-watermarks,stream-join,
  windowed-aggregation,streaming-inference}.md`; `docs/ml/inference/streaming.md`;
  `docs/examples/data-engineering/{incremental-ingest,cdc-pipeline,late-arriving-data}.md`.
- Skills: `write-a-batcher-pipeline` (relational core); `read-and-write-data` (sources/sinks);
  `manage-a-lakehouse-table` (Delta/Iceberg upserts — the idempotent sink);
  `validate-data-quality` (gating micro-batches); `run-a-distributed-job`;
  `build-an-ml-pipeline` (streaming inference); `debug-a-batcher-query`.
