# Late data and watermarks

A phone goes through a tunnel. It buffers events locally, comes back online eight minutes
later, and flushes them. Your hourly window for 00:00 closed at 01:10, because you set the
lateness to ten minutes and an event stamped 01:20 came through. The tunnel events land at
01:25 with an event time of 00:47. They belong to a window that no longer exists.

:::{warning}
Batcher drops them. Silently: no side output, no counter in `recent_progress()`. Your
hourly total is quietly, permanently 3% low, and nothing in the job's metrics says so.
:::

This page is about choosing that tradeoff on purpose instead of discovering it in a
reconciliation ticket.

## What the watermark is

The watermark is a single number: `max(event_time observed) - lateness`. It is a claim
that no event older than this will be accepted. It advances only when event time advances,
and it never goes backward.

It does two things, and they are the same thing seen from either side:

- a window whose end is at or below the watermark is **closed**, meaning finalized,
  emitted, evicted from state;
- a row whose event time is below the watermark is **late**, dropped before it is folded
  into any window.

`lateness` is the dial between those:

| `lateness` | Latency | State | Stragglers |
| --- | --- | --- | --- |
| Small | low: a window emits soon after it ends | small: few windows open at once | thrown away |
| Large | high: a result waits before it is emitted | large: every open window is retained | counted |

## Watch a row get dropped

Three micro-batches. The third carries a straggler: an event at 00:20, arriving after the
stream's event time has already jumped to 02:10.

:::{dropdown} The fixture: three micro-batches, and an hourly window over them

```python
import datetime as dt

import pyarrow as pa

import batcher as bt
from batcher import col

base = dt.datetime(2024, 1, 1)
minute = dt.timedelta(minutes=1)
schema = pa.schema([("ts", pa.timestamp("us")), ("amount", pa.int64())])


def feed():
    yield pa.record_batch({"ts": [base, base + 30 * minute], "amount": [3, 5]},
                          schema=schema)
    yield pa.record_batch({"ts": [base + 130 * minute], "amount": [1]}, schema=schema)
    yield pa.record_batch({"ts": [base + 20 * minute], "amount": [100]}, schema=schema)


def hourly(lateness):
    stream = bt.from_batches(feed, schema, bounded=False)
    windowed = (
        stream.with_watermark("ts", lateness)
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("amount").sum())
    )
    return [b.to_pydict() for b in windowed.iter_batches()]
```
:::

Same fixture, same query, one number changed:

::::{tab-set}
:::{tab-item} lateness = 10 minutes

With ten minutes of lateness, the second micro-batch (event time 02:10) pushes the
watermark to 02:00, which closes and emits the 00:00 window. The straggler that follows is
below the watermark:

```python
for emitted in hourly("10m"):
    print(emitted)
# {'w': [datetime.datetime(2024, 1, 1, 0, 0)], 'total': [8]}
# {'w': [datetime.datetime(2024, 1, 1, 2, 0)], 'total': [1]}
```

`total: 8`. The straggler's `100` is not in it, and never will be.
:::

:::{tab-item} lateness = 3 hours

The watermark after the 02:10 event is 23:10 of the previous day, which
closes nothing, so the 00:00 window is still open when the straggler arrives:

```python
for emitted in hourly("3h"):
    print(emitted)
# {'w': [datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 1, 2, 0)], 'total': [108, 1]}
```

`total: 108`. The straggler counted. Note what else changed: nothing was emitted *during*
the stream. Both windows came out in the end-of-stream flush, because the watermark never
got far enough to close either one. That is the cost, made concrete: a three-hour lateness
means an hourly window is three hours late to your dashboard, and its state sits in memory
the whole time.
:::
::::

## Picking the number

Measure it, don't guess it. Land the raw events first (see
[Kafka to the lake](kafka-etl.md)), then look at the distribution of
`processing_time - event_time` in the bronze table. The p99 of that lag is a defensible
lateness. The max is not: one pathological device should not hold every window open.

Two failure shapes to keep in mind:

**A stalled watermark.** Event time advances on the *maximum* event time seen. If a
partition goes idle, or a source stops producing, nothing advances, so no window closes and
state grows. Batcher does not let that end in an OOM: retained state is checked against
`memory.streaming_state_max_bytes` and a `ResourceError` names the column whose watermark
is not advancing. Read it as a diagnosis, not a budget request.

**A clock from the future.** One device with a badly-set clock emits an event stamped next
Tuesday. `max(event_time)` jumps to next Tuesday, the watermark jumps with it, and every
window you have is closed and emitted at once, while every subsequent real event is now
"late" and dropped. Sanitize event time at the edge; a `filter` on a plausible range is
cheap, and it runs in Rust:

```python
sane = bt.from_batches(feed, schema, bounded=False).filter(
    col("ts") < bt.lit(dt.datetime(2030, 1, 1))
)
print(sane.is_streaming)
# True
```

## Bounded state for deduplication

The same mechanism bounds a dedup. `drop_duplicates_within_watermark` keeps the first row
per key inside the watermark window and forgets keys the watermark has passed, so the
seen-key set does not grow forever. An at-least-once producer that re-sends on a timeout is
exactly what this is for:

```python
dedup_schema = pa.schema([
    ("id", pa.string()),
    ("ts", pa.timestamp("us")),
    ("v", pa.int64()),
])


def dupes():
    yield pa.record_batch({"id": ["x", "y"], "ts": [base, base], "v": [1, 2]},
                          schema=dedup_schema)
    yield pa.record_batch(
        {"id": ["x", "z"], "ts": [base + minute, base + minute], "v": [3, 4]},
        schema=dedup_schema,
    )


stream = bt.from_batches(dupes, dedup_schema, bounded=False)
unique = stream.drop_duplicates_within_watermark(["id"], event_time="ts", lateness="10m")
for batch in unique.iter_batches():
    print(batch.to_pydict()["id"])
# ['x', 'y']
# ['z']
```

The second `x` was dropped. Its retry arrived inside the watermark, which is the deal: a
duplicate that arrives *after* the watermark has passed its key will not be caught, because
the key is gone.

:::{important}
Your lateness is therefore also your deduplication window. Size it to
your producer's retry behavior, not only to your window latency.
:::

## What you don't get

:::{important}
There is no late-data side output. A dropped row is not routed anywhere, and
`StreamingQueryProgress` reports `num_input_rows` and `num_output_rows` but no late-row
count. If losing a straggler is unacceptable for your use case, the honest options are: set
a lateness that actually covers your lag distribution; or land raw events and recompute the
affected windows in a batch job, which is a reconciliation pipeline, not a streaming one.
:::

## See also

- [Windowed aggregation](windowed-aggregation.md): the windows this watermark closes.
- [Joining two streams](stream-join.md): the same watermark, evicting join buffers.
- [Kafka to the lake](kafka-etl.md): landing the raw events you measure the lag distribution
  on.
- [Streaming](../../user-guide/streaming.md): watermarks, triggers, and output modes in full.
- [Late-arriving data](../data-engineering/late-arriving-data.md): the batch reconciliation
  pipeline this page keeps pointing at.
- [Deduplication](../data-engineering/deduplication.md): dedup without a watermark to bound it.
- [Data quality](../../user-guide/data-quality.md): catching the clock-from-the-future row at
  the edge.
- [Spilling](../../deep-dives/spilling.md): what bounded state buys you, and where it goes when
  it does not fit.
