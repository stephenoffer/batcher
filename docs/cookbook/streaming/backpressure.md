# Backpressure

This page covers how a streaming query stops its source from handing it more than it can
process, and how to tell whether it is working.

## What runs away

A micro-batch that overruns its trigger interval leaves the next one starting late, against a
backlog that grew while it ran. The next one overruns by more. Nothing about this is gradual:
each batch is larger than the last, so the divergence compounds, and it does not end in a slow
query. It ends in the epoch that no longer fits in memory.

The signal is in the progress record from the start, which is why it is worth watching before
anything is on fire:

```python
# docs: skip
for p in query.recent_progress:
    print(p.batch_id, p.num_input_rows, p.duration_ms, p.behind_by_ms)
```

`behind_by_ms` is how far past its trigger cadence the batch ran. A query that is keeping up
reports zero. A query in the runaway reports a number that grows every batch.

## The static cap

Every unbounded source accepts a per-trigger bound, under the names a Spark job already uses:

```python
# docs: skip
import batcher as bt

events = bt.read.kafka(
    "orders",
    bootstrap_servers="broker:9092",
    max_offsets_per_trigger=50_000,
)
```

| Source | Options |
| --- | --- |
| Kafka, Kinesis, Pulsar, Pub/Sub, Event Hubs | `max_offsets_per_trigger`, `max_bytes_per_trigger` |
| Incremental file discovery (Auto Loader) | `max_files_per_trigger`, `max_bytes_per_trigger` |

Both bounds apply together, and whichever trips first wins. Capping bytes as well as count
matters more than it looks: ten thousand 4 KiB JSON files and three 8 GiB Parquet files are
both "a backlog", and only one of them fits.

This is a real bound and it is the right first move. Its limit is that you have to pick the
number, and you have to pick it for the worst trigger the query will ever see — so it throttles
every other one, and it goes stale as soon as the cluster, the data, or the plan changes.

## The adaptive cap

Turning on `streaming.backpressure_enabled` derives the cap instead, from the query's own
measured throughput:

```python
# docs: skip
import batcher as bt

with bt.config_context(bt.Config(
    streaming=bt.StreamingConfig(backpressure_enabled=True),
)):
    query = events.write_stream.delta("s3://lake/orders", trigger=bt.Trigger.processing_time(30))
```

Each completed micro-batch reports how many rows it consumed and how long it took, which is a
processing rate. The controller compares that against the rate the source is currently being
admitted at and adjusts. It is a PID controller, with Spark's knob names and Spark's defaults,
so tuning advice written for `spark.streaming.backpressure.pid.*` carries over unchanged.

| Option | Default | What it weights |
| --- | --- | --- |
| `backpressure_enabled` | `False` | Whether the controller runs at all. |
| `backpressure_pid_proportional` | `1.0` | The current error, so a query that suddenly slows is throttled on the next trigger. |
| `backpressure_pid_integral` | `0.2` | The accumulated backlog. |
| `backpressure_pid_derivative` | `0.0` | The error's rate of change, damping overshoot on a bursty source. |
| `backpressure_min_rate` | `100.0` | Rows per second the derived rate can never fall below. |
| `backpressure_max_rows_per_trigger` | `0` | A hard ceiling on the derived cap, independent of the source. `0` is unbounded. |

The integral term is the one worth understanding before changing. It removes *steady-state*
error: a purely proportional controller settles at a rate slightly above what the query
sustains and then stays permanently a little behind, which is the compounding case again,
arrived at more slowly.

### What it will not do

The controller only ever *lowers*. Whatever `max_offsets_per_trigger` you configured remains
the ceiling, so enabling this can tighten a static cap and never loosen one.

It abstains rather than guessing. It names no cap for the first micro-batch, whose rate
measures a cold cache and a connection handshake rather than the steady state, and it ignores
a trigger that consumed no rows. An empty batch's rate is zero, and acting on that would drive
a healthy stream to the floor the first time its topic went quiet, then keep it there, because
the throttle shrinks the next batch too.

The derived rate can never reach zero. A trigger admitting nothing publishes no progress
record, and a controller with no progress can never revise the cap that stalled it.

An admission cap changes how much of a stream a trigger reads. It never changes what the query
computes from the rows it read, so none of this can change a result.

:::{note}
The controller acts through a source's per-trigger admission, so it paces broker sources —
Kafka, Kinesis, Pulsar, Pub/Sub, Event Hubs. A file source has no per-trigger row admission to
narrow, and is governed by `max_files_per_trigger` and `max_bytes_per_trigger` instead. This
matches where Spark's rate controller applies.
:::

## The other backpressure

Rate control paces the *source*. It is not the only backpressure in a distributed streaming
query, and the two do not substitute for each other.

A shuffle channel is paced by a credit window: one credit is one in-flight batch slot, and a
producer that runs its peer out of credits blocks until the consumer grants more. That is what
stops a fast mapper from flooding a slow reducer, and no rate limit on the source can do it.
Equally, no credit window can stop a trigger from reading a backlog that will not fit.

Underneath, both are the same identity. Little's Law says work in progress equals arrival rate
times time in system, `L = lambda W`. The credit window measures `lambda` in bytes per second
and `W` as a round trip, giving the bandwidth-delay product. The rate controller measures
`lambda` in rows per second and `W` as the trigger interval, giving a per-trigger row cap. Two
controllers, one law, which is why they compose rather than fight for the same memory.

See {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`
for how the window is sized and adapted.

## Verifying it

Watch `behind_by_ms` across the recent progress. A query under working backpressure holds it
near zero with `num_input_rows` varying; a query without it shows both climbing together.

```python
# docs: skip
recent = query.recent_progress[-10:]
print([p.behind_by_ms for p in recent])
print([p.num_input_rows for p in recent])
```

If `behind_by_ms` is climbing while `num_input_rows` is flat at your configured cap, the
static cap is already binding and the controller has no room to help. Lower it, or find the
cost in `duration_ms_map` first: a batch whose time is in `walCommit` is a slow checkpoint, not
a slow query, and throttling the source will not fix it.

## Requirements and limitations

- Adaptive backpressure is off by default, matching Spark. A controller acting on an estimate
  it has not earned is worse than none.
- It paces broker sources. File sources use the static per-trigger caps.
- It needs at least two completed micro-batches before it names a cap.
- It cannot help a query whose bottleneck is the sink or the checkpoint. Check
  `duration_ms_map` before reaching for it.

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: sources, sinks, triggers, output modes.
- {doc}`Monitoring </user-guide/moving-data/streaming-monitoring>`: the full progress record.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`:
  the shuffle's half of the same discipline.
- {doc}`Kafka ETL </cookbook/streaming/kafka-etl>`: the source options in a working pipeline.
