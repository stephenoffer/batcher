# Streaming

The same operators, over a source that never ends.

Two things change when the input is unbounded. Time becomes something you have to reason
about, because "the last hour" is not a fact you can look up. And the job *will* be
restarted, so every recipe here has to answer what happens when it is.

Batch is the bounded special case of streaming in this engine: the same operators, the same
windows, the same plan. So each page shows the logic on a small bounded table you can
actually run, then the unbounded wiring around it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Kafka ETL
:link: kafka-etl
:link-type: doc
Read a topic, clean it, land it.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Windowed aggregation
:link: windowed-aggregation
:link-type: doc
The only thing that makes a streaming group finishable.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Stream join
:link: stream-join
:link-type: doc
The join that buffers forever if you let it.
:::

:::{grid-item-card} {octicon}`check;1.1em` Exactly-once sink
:link: exactly-once-sink
:link-type: doc
At-least-once delivery plus a duplicate is a wrong answer.
:::

:::{grid-item-card} {octicon}`alert;1.1em` Late data and watermarks
:link: late-data-watermarks
:link-type: doc
What a dropped straggler costs, and where it goes.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Streaming inference
:link: streaming-inference
:link-type: doc
A resident model over micro-batches, and the poison record.
:::
::::

## The sharp edges, up front

Streaming in Batcher has real limits, and each one is the kind you would rather read about
than discover. None of them is a bug you can configure away.

| The edge | What actually happens | Where it is covered |
| --- | --- | --- |
| A stream-stream join has no checkpoint | It writes to a sink like any other streaming query, but `checkpoint=` is refused: the join's state is two buffers and two watermarks, not a source offset, so a restart begins with an empty join. | {doc}`Stream join </cookbook/streaming/stream-join>` |
| A stream joined to a static dimension table serves a snapshot | The table is read once, when the query starts, and never refreshed. Restart the query to pick up a new one. | {doc}`Stream join </cookbook/streaming/stream-join>` |
| The file sink dedups by *position* | A different batch 0 written into a directory that already holds `part-batch00000.parquet` is skipped, silently. One query, one output directory. | {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>` |
| A non-replayable source ignores `checkpoint=` | No offsets are recorded, so a restart re-reads from the beginning. Nothing warns you. | {doc}`Exactly-once sink </cookbook/streaming/exactly-once-sink>` |
| Late rows are dropped, not diverted | No side output and no dead-letter: the rows are gone. `num_late_rows` on each micro-batch counts them, so the shortfall is visible. | {doc}`Late data and watermarks </cookbook/streaming/late-data-watermarks>` |

## See also

- {doc}`Streaming </user-guide/moving-data/streaming>`: the full source, sink, trigger, and output-mode
  reference.
- {doc}`Kafka integration </integrations/streams/kafka>`: brokers, consumer groups, and splits.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: the batch write surface and Delta commits.
- {doc}`Late-arriving data </cookbook/data-engineering/ingest/late-arriving-data>`: the batch reconciliation
  half of the same problem.
- {doc}`ML recipes </cookbook/ml/pipelines/index>`: the model stages that these queries run per micro-batch.

```{toctree}
:hidden:

kafka-etl
windowed-aggregation
stream-join
exactly-once-sink
late-data-watermarks
streaming-inference
```
