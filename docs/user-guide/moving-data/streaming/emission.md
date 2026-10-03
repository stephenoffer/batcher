# When a streaming query emits

This page covers *when* each relational shape produces output over an unbounded source, and
what to do about the shapes that produce none until the input ends. It assumes the pipeline
basics in {doc}`/user-guide/moving-data/streaming/index`.

Every relational shape that runs on a bounded dataset also runs on an unbounded one. What differs is *when* it produces output, and over a source that never ends, that decides whether you see anything at all.

```python
import datetime as dt

import pyarrow as pa

import batcher as bt
from batcher import col

schema = pa.schema([("user", pa.string()), ("amount", pa.int64()), ("ts", pa.timestamp("us"))])


def at(h, m):
    return dt.datetime(2024, 1, 1, h, m)


def feed():
    yield pa.record_batch({"user": ["a", "b"], "amount": [10, 5], "ts": [at(10, 5), at(10, 40)]}, schema=schema)
    yield pa.record_batch({"user": ["a", "c"], "amount": [7, 3], "ts": [at(11, 25), at(11, 50)]}, schema=schema)


stream = bt.from_batches(feed, schema, bounded=False)

# Row-wise shapes emit per batch, as rows arrive.
print([b.num_rows for b in stream.filter(col("amount") > 4).iter_batches()])
# [2, 1]
```

The shapes fall into two groups:

| Shape | When it emits |
|-------|---------------|
| `filter` / `select` / `with_columns` / `map_batches` | Per window, as rows arrive. |
| `limit(n)` | As rows arrive, and stops reading at `n`. |
| `distinct().limit(n)` | As rows arrive, and stops reading at `n` distinct rows. |
| `with_watermark(...)` + a `window(...)` group key | Per window, as the watermark closes each one. |
| `drop_duplicates_within_watermark(...)` | Per batch. The seen-key set is watermark-bounded. |
| A stream-static join, a `join_stream` interval join, a session window | Per batch. |
| `group_by(...).agg(...)` with no watermark | Once, at end of input. |
| `distinct()` with no cap | Once, at end of input. |
| `sort(...).limit(n)` (top-N) | Once, at end of input. |
| `sort(...)` with no limit | Refused: it cannot bound its memory either. |

This table is about {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`. A
*materializing* terminal such as `to_pydict()` is stricter, because it has to
return one finished result: it refuses a top-N or a keyed `distinct(subset=...)` over a
stream outright. {doc}`index` has the detail, under "Look at a stream before you
build on it".

The "once, at end of input" shapes fold everything into one running state and finalize when the input stops. That works for a source that drains, such as an incremental file read under {py:meth}`Trigger.available_now() <batcher.Trigger.available_now>`. Over a Kafka topic, end of input never arrives, so `iter_batches()` warns with a `PerformanceWarning` before its first read and names the ways to get output sooner.

Side by side on the same four triggers, the difference is entirely timing:

![A grid of three shapes against four triggers of one unbounded stream, plus a final column for when the input ends, with illustrative event times, a one-hour window, and ten minutes of allowed lateness. The highest event time seen at each trigger is 10:40, 11:05, 11:25 and 11:50, so the watermark is 10:30, 10:55, 11:15 and 11:40. A row-wise shape such as filter, select, with_columns or map_batches emits each trigger's own rows as they arrive, and has nothing left to emit when the input ends. A with_watermark plus window aggregate emits nothing on triggers 1 and 2, emits the 10:00 to 11:00 window on trigger 3 because 11:15 is the first watermark at or past that window's end, and emits nothing on trigger 4 because the 11:00 to 12:00 window is still open; that window is emitted when the input ends. A group_by().agg() with no watermark emits nothing on any trigger and emits the whole result only when the input ends. A draining source reaches that last column. A Kafka topic never does.](/_static/diagrams/streaming_emission.svg)

Memory isn't the issue. A top-N keeps only the best `n` rows, so it is bounded, yet it has no answer until the input ends. Neither does a global `sum`.

To get output as rows arrive, ask a question that has an answer so far. Window it, so the watermark closes each finite window:

```python
windowed = (
    stream.with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1 hour"))
    .agg(total=col("amount").sum())
)
for batch in windowed.iter_batches():
    print(batch.to_pydict()["total"])
# [15]
# [10]
```

The 10:00 window emits as soon as the watermark passes 11:00. The 11:00 window is still open when the input ends, so it emits then. Or run it as a streaming query, which emits the running result on every trigger:

```python
# docs: skip
# A streaming query, which emits the running result every trigger.
q = (
    stream.group_by("user")
    .agg(total=col("amount").sum())
    .write(
        "out/totals",
        format="parquet",
        output_mode="update",
        trigger=bt.Trigger.processing_time("30 seconds"),
        checkpoint="out/_ck",
    )
)
```

`output_mode="update"` emits the groups that changed this trigger and `"complete"` emits
every group every trigger. {doc}`index` covers both under "Output modes".

## What a streaming aggregate emits above itself

Row-wise operators above a streaming aggregate run on each snapshot the fold emits: a projection, a `filter` playing SQL's `HAVING`, and expressions over aggregates such as `col("v").sum() / bt.count()`. The answer matches the batch plan, on one machine or with `distributed=True`.

```python
schema2 = pa.schema([("user", pa.string()), ("amount", pa.int64())])


def feed2():
    yield pa.record_batch({"user": ["a", "b"], "amount": [10, 5]}, schema=schema2)
    yield pa.record_batch({"user": ["a"], "amount": [7]}, schema=schema2)


query = (
    bt.from_batches(feed2, schema2, bounded=False)
    .group_by("user")
    .agg(total=col("amount").sum(), n=bt.count())
    .with_columns(mean=col("total") / col("n"))
    .filter(col("total") > 5)
    .select("user", "mean")
    .write.memory("per_user_mean", trigger=bt.Trigger.available_now(), output_mode="complete")
)
query.await_termination()
print(sorted(bt.read_memory("per_user_mean").to_pydict()["user"]))
# ['a']
```

:::{dropdown} A `HAVING` filter that drops a group later
A `filter` above the aggregate can drop a group it kept on an earlier trigger, such as when a running `sum` falls back below a threshold. In `"complete"` mode each snapshot replaces the sink's contents, so the sink ends on the batch answer. In `"update"` mode Batcher emits only rows that are present and changed, with no tombstone record, so an upsert sink keeps that group's last emitted row. When a group can cross the threshold in both directions, use `"complete"`, or apply the threshold downstream of an `"update"` sink.
:::

`sort` and `limit` aren't row-wise, so sort downstream of the sink. `output_mode="append"` on a windowed aggregation refuses rather than approximating: a closed window is emitted once and never revised, so no later snapshot can correct a projection applied to a partial one. Use `"complete"` or `"update"`, or derive the columns downstream.

## First-row latency on a `map_batches` stream

The streaming iterator collects source batches into a window before calling a `map_batches` function, so the function can spread across the worker pool. The window closes on size (four million rows or 128 MiB) or on age, whichever comes first. The age bound, `streaming.max_window_latency_seconds`, defaults to one second and keeps a low-rate stream responsive. Raise it to trade first-row latency for larger windows. It applies only to unbounded sources. See {doc}`/configuration/options`.

## See also

- {doc}`/user-guide/moving-data/streaming/index`: the streaming surface these shapes are written against.
- {doc}`/user-guide/moving-data/streaming/stateful`: the watermark-bounded operators in depth.
- {doc}`/configuration/options`: `streaming.max_window_latency_seconds` and the rest of the cadence knobs.
