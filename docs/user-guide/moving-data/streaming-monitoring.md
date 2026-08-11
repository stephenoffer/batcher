# Monitoring a streaming query

This page describes how to watch a running streaming query: the handle it hands back, the
per-micro-batch metrics on it, the state its stateful operators are holding, and how to be
told about each batch instead of polling for it.

A streaming query runs for as long as you let it, so the two questions that matter are
whether it is keeping up and whether it is quietly dropping rows. Throughput answers
neither. The fields below do.

## Managing a query

`start`-style writes return a `StreamingQuery`:

```python
# docs: skip
q = clicks.write("s3://bucket/out", format="parquet",
                 trigger=bt.Trigger.processing_time("10 seconds"),
                 checkpoint="s3://bucket/_ckpt")
q.is_active            # True while running
q.status               # a point-in-time StreamingQueryStatus
q.recent_progress    # per-micro-batch metrics
q.exception()          # the failure that stopped it, or None (does not re-raise)
q.explain()            # the plan this query is running
q.process_all_available()  # block until the current backlog is done
q.stop()               # halt at the next micro-batch boundary
bt.streams()           # all active streaming queries
```

{py:meth}`explain <batcher.StreamingQuery.explain>` shows the *planned* tree only. `Dataset.explain(analyze=True)` runs the
query to measure it, which a stream cannot do twice: the source has moved on, and running
it again would double-consume the topic. Per-micro-batch measurements live in
`recent_progress` instead.

With several queries running, {py:func}`bt.await_any_termination(timeout=None) <batcher.await_any_termination>` blocks until
the first of them stops, re-raising its exception if it failed. This is the Spark
`awaitAnyTermination` pattern, for a driver that supervises multiple streams.

## Is the query keeping up?

Throughput tells you how fast a micro-batch ran. It cannot tell you whether that was fast
enough, because "enough" is the trigger interval. Each {py:class}`StreamingQueryProgress <batcher.StreamingQueryProgress>` carries
`behind_by_ms`, the milliseconds the batch overran its cadence, and `is_behind` for the
common check:

```python
# docs: skip
late = [p for p in q.recent_progress if p.is_behind]
if late:
    print(f"{len(late)} of the last {len(q.recent_progress)} batches ran long")
    print(max(p.behind_by_ms for p in late), "ms worst case")
```

A batch that occasionally runs long is normal. A `behind_by_ms` that grows batch over batch
means the query is falling behind its source, and the fix is upstream of the metric: a
larger trigger interval, more workers, or less work per row. Both fields are `0` for a
trigger with no interval (`once`, `available_now`, `continuous`), where there is no cadence
to be late for.

## Is it dropping rows, and how much state is it holding?

Falling behind is one of the two ways a streaming query goes wrong. The other is dropping
rows quietly. A windowed aggregation discards every row that arrives below its watermark,
which is correct and produces a total that is simply short. `num_late_rows` on each
micro-batch is the count of what it discarded:

```python
# docs: skip
dropped = sum(p.num_late_rows for p in q.recent_progress)
if dropped:
    print(f"{dropped} rows arrived too late for their window")
```

A non-zero count means the allowed lateness is tighter than the stream's real skew. Widen
it with {py:meth}`with_watermark(col, "30 minutes") <batcher.Dataset.with_watermark>` and the rows are counted at the cost of holding
more windows open.

`state_operators` is that other side: one {py:class}`StateOperatorProgress <batcher.StateOperatorProgress>` per stateful operator,
carrying the rows it retains, what the last batch evicted, the bytes it holds, and its
watermark.

```python
# docs: skip
for op in q.last_progress.state_operators:
    print(op)
# windowed_aggregate: 412 rows retained (26384 bytes), 37 evicted
```

Rows retained that only ever grow, with nothing evicted, is the shape that ends in a
{py:exc}`ResourceError <batcher.ResourceError>` hours later: the watermark is not advancing, or the grouping key is too
wide. `event_time_watermark_micros` on the progress record is the minimum watermark across
those operators, which is how far the query as a whole has advanced through event time.

Two more fields say where the rows came from and where they went. `sources` is a tuple of
{py:class}`SourceProgress <batcher.SourceProgress>`, one per input, carrying the rows it produced and the offsets the batch
moved between. `sink` is a {py:class}`SinkProgress <batcher.SinkProgress>` with the rows written and the receipt the sink
returned, which is the same token the checkpoint's commit log records:

```python
# docs: skip
p = q.last_progress
print(p.sources[0].description, p.sources[0].num_input_rows, p.sources[0].end_offset)
print(p.sink.description, p.sink.num_output_rows, p.sink.token)
```

### What a multi-source or row-retaining query reports

A query whose plan is a stream-stream join, a stream-static join, a session window, a
watermark dedup, a limit, or a union of streams runs through a driver that produces
finished rows and hands them to the sink. The engine sees what the driver hands it, not
what the driver read, so for these queries:

- `num_input_rows` counts the rows the driver emitted, not the rows it consumed from the
  source.
- `state_operators` is empty, even though these operators do retain state.

The state is still bounded, and still raises a {py:exc}`ResourceError <batcher.ResourceError>` naming the stall when a
watermark stops advancing. What is missing is the per-batch reporting, not the guard.

## Reacting to batches as they happen

Polling `recent_progress` on a timer misses batches, because the window is bounded, and
cannot see a query start at all. Register a {py:class}`StreamingQueryListener <batcher.StreamingQueryListener>` and each event arrives
once, as it happens:

```python
import batcher as bt

class LatenessAlarm(bt.StreamingQueryListener):
    def on_query_started(self, event):
        print(f"{event.name} started")

    def on_query_progress(self, event):
        if event.progress.num_late_rows:
            print(f"{event.name}: {event.progress.num_late_rows} late rows")

    def on_query_terminated(self, event):
        if event.exception:
            print(f"{event.name} failed: {event.exception}")

alarm = LatenessAlarm()
bt.add_streaming_listener(alarm)
print(len(bt.streaming_listeners()))
# 1
print(bt.remove_streaming_listener(alarm))
# True
```

Every query in the process reports to every registered listener, whether it started before
or after the registration. Override only the callbacks you need; the rest do nothing. The
PySpark spellings (`onQueryStarted`, `onQueryProgress`, `onQueryTerminated`) work too, so a
ported listener keeps working.

Each callback receives one event type. A {py:class}`QueryStartedEvent <batcher.QueryStartedEvent>` carries the query's `name`,
`id`, and start `timestamp`. A {py:class}`QueryProgressEvent <batcher.QueryProgressEvent>` carries `name` and the
`StreamingQueryProgress` for the batch that just finished. A {py:class}`QueryTerminatedEvent <batcher.QueryTerminatedEvent>` carries
`name`, `id`, and `exception`, which is `None` for a clean stop and the formatted failure
otherwise, as a string so it can be shipped to a metrics system without pickling.

:::{warning}
Callbacks run on the query's own loop thread, between micro-batches. Work done there is
latency the next batch pays, so push to a queue rather than making a network call inline.
A listener that raises is logged and skipped, never allowed to fail the query.
:::

## What an idle stream does

A stream with nothing to read is idle, not finished. A query keeps running until you stop
it, its source is genuinely bounded, or it was started with a draining trigger. Between
looks at an empty source it waits `streaming.idle_poll_seconds`, so a quiet stream costs
one listing every fraction of a second rather than a spinning core:

```python
import batcher as bt
from batcher import Config, StreamingConfig

# Look for new data more eagerly, for a latency-sensitive stream.
cfg = Config().replace(streaming=StreamingConfig(idle_poll_seconds=0.05))
print(cfg.streaming.idle_poll_seconds)
# 0.05
```

{py:class}`StreamingConfig <batcher.StreamingConfig>` also bounds how much history a handle keeps: `recent_progress` holds the
last `progress_history` micro-batches, not every batch since the query started.

Apply it for one query with {py:func}`bt.config_context(cfg) <batcher.config_context>`, or process-wide with
{py:func}`bt.set_config(cfg) <batcher.set_config>`.


## Shipping progress somewhere else

A progress record's destination is usually a log line or a metrics system, and both want
data rather than a dataclass. `to_dict()` and `json()` produce Spark's shape, keyed in its
camelCase, so a dashboard written against `StreamingQueryProgress` reads these unchanged:

```python
# docs: skip
for p in q.recent_progress:
    metrics.emit(p.to_dict())          # or p.json() straight into a log line
```

`durationMs` in that payload is where the micro-batch's time went: `latestOffset` (asking
the source what is available), `addBatch` (running the plan and writing the sink),
`walCommit` (the offset and commit log fsyncs), and `triggerExecution` for the whole batch.
A total alone cannot tell a slow query from a slow *checkpoint*, and those have opposite
remedies.

## Identifying a query across restarts

`q.id` is stable across restarts of the same query; `q.run_id` is fresh on every start.
Keying a dashboard only on `id` cannot tell a query that has been up for a week from one
that has crash-looped every ten minutes. Both have Spark's camelCase spellings too.

A supervisor that loops on `bt.await_any_termination()` should call `bt.reset_terminated()`
after handling one, exactly as in Spark: the call returns immediately once any query has
terminated and keeps doing so until the record is cleared, so a loop that forgets spins.

## See also

- {doc}`streaming`: sources, sinks, triggers, output modes, and checkpoints.
- {doc}`/cookbook/streaming/late-data-watermarks`: what the allowed lateness costs.
- {doc}`/configuration/options`: the `streaming` section's idle cadence and history bound.
- {doc}`/user-guide/operate/running/metrics`: the same per-batch numbers as scrapeable counters, so a chart sees the lag a single reading cannot.
