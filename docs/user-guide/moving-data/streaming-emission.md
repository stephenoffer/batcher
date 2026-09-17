# When a streaming query emits

This page covers *when* each relational shape produces output over an unbounded source, and
what to do about the shapes that produce none until the input ends. It assumes the pipeline
basics in {doc}`/user-guide/moving-data/streaming`.

Every relational shape that works on a bounded dataset also runs on an unbounded one, and
`iter_batches()` will drive all of them. What differs is *when* a shape produces output,
and over a source that never ends that difference decides whether you see anything at all.

The shapes fall into two groups, and the split isn't about memory:

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
stream outright. {doc}`streaming` has the detail, under "Look at a stream before you
build on it".

The second group folds its input into one running state and finalizes when the input
stops. That is the right answer for a source that ends, including an unbounded-by-type
source that drains, such as an incremental file read under
{py:meth}`Trigger.available_now() <batcher.Trigger.available_now>`. Over a source that
genuinely never ends, such as a Kafka topic, "at end of input" never arrives and the query
consumes without emitting.

Put one shape from each row of the table side by side on the same four triggers and the difference is entirely one of timing:

![A grid of three shapes against four triggers of one unbounded stream, plus a final column for when the input ends, with illustrative event times, a one-hour window, and ten minutes of allowed lateness. The highest event time seen at each trigger is 10:40, 11:05, 11:25 and 11:50, so the watermark is 10:30, 10:55, 11:15 and 11:40. A row-wise shape such as filter, select, with_columns or map_batches emits each trigger's own rows as they arrive, and has nothing left to emit when the input ends. A with_watermark plus window aggregate emits nothing on triggers 1 and 2, emits the 10:00 to 11:00 window on trigger 3 because 11:15 is the first watermark at or past that window's end, and emits nothing on trigger 4 because the 11:00 to 12:00 window is still open; that window is emitted when the input ends. A group_by().agg() with no watermark emits nothing on any trigger and emits the whole result only when the input ends. A draining source reaches that last column. A Kafka topic never does.](/_static/diagrams/streaming_emission.svg)

Memory is not the signal to watch here, and top-N is the case that shows why: it keeps only
the running best `n` rows, so it is perfectly bounded and still produces nothing until the
input ends. A global `sum` is the same. Neither has an answer while rows are arriving,
which is the actual reason, and it is a property of the question rather than of the engine.

To get output as rows arrive from one of those shapes, ask a question that has an answer so
far. Either window it, so each window is finite and the watermark closes it, or run it as a
streaming query, which emits the running result on the trigger:

```python
# docs: skip
# Windowed: each window is a finite question, closed by the watermark.
(
    stream.with_watermark("ts", "10 minutes")
    .group_by(w=bt.window(col("ts"), "1 hour"))
    .agg(total=col("amount").sum())
    .iter_batches()
)

# Or a streaming query, which emits the running result every trigger.
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
every group every trigger. {doc}`streaming` covers both under "Output modes".

## What a streaming aggregate emits above itself

A streaming aggregate is not limited to a bare `group_by(...).agg(...)`. The row-wise
operators above it run too, applied to each snapshot the fold emits: a projection, a
`filter` playing the part of SQL's `HAVING`, and any expression written *over* aggregates.
The last is the common case, because `col("v").sum() / bt.count()` is one keyword to you
and a projection over an aggregate to the engine, as are `col("v").max() - col("v").min()`
and the regression and correlation functions.

The answer matches the batch plan, because a row-wise operator's output for a row depends
on that row alone, so applying it to the running result is what applying it to the whole
input computes. This holds on a cluster as well: the driver applies the same operators to
the combined result, so `distributed=True` returns what one machine returns.

```python
import pyarrow as pa

import batcher as bt
from batcher import col

schema = pa.schema([("user", pa.string()), ("amount", pa.int64())])


def feed():
    yield pa.record_batch({"user": ["a", "b"], "amount": [10, 5]}, schema=schema)
    yield pa.record_batch({"user": ["a"], "amount": [7]}, schema=schema)


query = (
    bt.from_batches(feed, schema, bounded=False)
    .group_by("user")
    .agg(total=col("amount").sum(), n=bt.count())
    .with_columns(mean=col("total") / col("n"))
    .filter(col("total") > 5)
    .select("user", "mean")
    .write.memory("per_user_mean", trigger=bt.Trigger.available_now(), output_mode="complete")
)
query.await_termination()
print(sorted(bt.read_memory("per_user_mean").to_pydict()["user"]))
```

`sort` and `limit` stay out. Neither is row-wise, so neither has a meaning on a *running*
result that matches what it means over the whole input. Sort downstream of the sink.

One shape refuses rather than approximating: `output_mode="append"` on a windowed
aggregation. A closed window is emitted once and never revised, so no later snapshot can
correct a projection applied to a partial one. Use `"complete"` or `"update"`, which
re-emit, or derive the columns downstream of the sink.

## First-row latency on a `map_batches` stream

A `map_batches` function only spreads across the worker pool when it is handed several
batches at once, so the streaming iterator collects source batches into a window before
calling it. That window closes on size or on age, whichever comes first. The age bound is
`streaming.max_window_latency_seconds`, one second by default, and it is what keeps a
low-rate stream responsive: without it the window would wait for four million rows or
128 MiB, which at 2,000 rows a second is 33 minutes before the first output and on a slower
topic considerably longer. Raise it to trade first-row latency for larger, more efficient
windows. It applies only to unbounded sources, so batch reads keep the size-based window
and their existing throughput unchanged. See {doc}`/configuration/options`.

## See also

- {doc}`/user-guide/moving-data/streaming`: the streaming surface these shapes are written against.
- {doc}`/user-guide/moving-data/streaming-stateful`: the watermark-bounded operators in depth.
- {doc}`/configuration/options`: `streaming.max_window_latency_seconds` and the rest of the cadence knobs.
