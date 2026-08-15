# When a streaming query emits

This page covers *when* each relational shape produces output over an unbounded source, and
what to do about the shapes that produce none until the input ends. It assumes the pipeline
basics in {doc}`/user-guide/moving-data/streaming`.

Every relational shape that works on a bounded dataset also runs on an unbounded one, and
`iter_batches()` will drive all of them. What differs is *when* a shape produces output,
and over a source that never ends that difference decides whether you see anything at all.

Two groups, and the split is not about memory:

| Shape | When it emits |
|-------|---------------|
| `filter` / `select` / `with_columns` / `map_batches` | Per window, as rows arrive. |
| `limit(n)` | As rows arrive, and stops reading at `n`. |
| `distinct().limit(n)` | As rows arrive, and stops reading at `n` distinct rows. |
| `with_watermark(...)` + a `window(...)` group key | Per window, as the watermark closes each one. |
| `drop_duplicates_within_watermark(...)` | Per batch; the seen-key set is watermark-bounded. |
| A stream-static join, a `join_stream` interval join, a session window | Per batch. |
| `group_by(...).agg(...)` with no watermark | Once, at end of input. |
| `distinct()` with no cap | Once, at end of input. |
| `sort(...).limit(n)` (top-N) | Once, at end of input. |
| `sort(...)` with no limit | Refused: it cannot bound its memory either. |

This table is about {py:meth}`iter_batches() <batcher.Dataset.iter_batches>`. A
*materializing* terminal such as `head()` or `to_pydict()` is stricter, because it has to
return one finished result: it refuses a top-N or a keyed `distinct(subset=...)` over a
stream outright. See "Looking at a stream before you build on it" below.

The second group folds its input into one running state and finalizes when the input
stops. That is the right answer for a source that ends, including an unbounded-by-type
source that drains, such as an incremental file read under
{py:meth}`Trigger.available_now() <batcher.Trigger.available_now>`. Over a source that
genuinely never ends, such as a Kafka topic, "at end of input" never arrives and the query
consumes without emitting.

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
(stream.with_watermark("ts", "10 minutes")
       .group_by(w=bt.window(col("ts"), "1 hour"))
       .agg(total=col("amount").sum())
       .iter_batches())

# Or a streaming query, which emits the running result every trigger.
q = (stream.group_by("user").agg(total=col("amount").sum())
           .write("out/totals", format="parquet",
                  output_mode="update",
                  trigger=bt.Trigger.processing_time("30 seconds"),
                  checkpoint="out/_ck"))
```

`output_mode="update"` emits the groups that changed this trigger and `"complete"` emits
every group every trigger. Both are covered in the "Output modes" section below.

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
