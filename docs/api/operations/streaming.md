# Streaming API

This page is the reference for Batcher's streaming surface: the trigger and output-mode
values a streaming write takes, what a running query reports, and the listener interface
that receives those reports as they happen.

For how to use them, see {doc}`/user-guide/moving-data/streaming/index` and
{doc}`/user-guide/moving-data/streaming/monitoring`. The handle a streaming write returns
is documented with the rest of the {py:class}`Dataset <batcher.Dataset>` surface in {doc}`/api/symbols/dataset-terminal`.

## Triggers and output modes

A trigger says when the engine fires a micro-batch. An output mode says how that batch's
result reaches the sink. Both spell their values the way Spark does.

```{eval-rst}
.. autoclass:: batcher.Trigger
   :members:

.. autoclass:: batcher.OutputMode
   :members:
```

The three modes differ only in what each one emits from the same input, so the figure runs
one input through all three:

![What append, complete and update each emit for one sequence of micro-batches. Three batches of key-value pairs feed a grouped max. Append emits each batch's own rows and is legal only for a pipeline with no aggregate: make_processor raises at start() for append over an unwindowed aggregate, which needs a watermark and a windowed group key. Complete emits the whole running result on every trigger, including a third trigger that changed nothing. Update anti-joins the new result against the one it last emitted, over every column, so a group whose value did not move is not re-sent and that third trigger emits no rows at all. The sink adds its own restriction: a path or Delta sink accepts append only, so complete and update need a memory sink or for_each_batch.](/_static/diagrams/output_modes.svg)

## Query progress and status

A running query emits one `StreamingQueryProgress` record per completed micro-batch. The
record carries that batch's row counts and duration, what each source contributed and the
sink accepted, and one `StateOperatorProgress` per stateful operator holding state. Late
data shows up there, as `num_late_inputs_dropped`: the inputs that arrived behind the
watermark and were discarded.

```{eval-rst}
.. autoclass:: batcher.StreamingQueryProgress
   :members:

.. autoclass:: batcher.StreamingQueryStatus
   :members:

.. autoclass:: batcher.StateOperatorProgress
   :members:

.. autoclass:: batcher.SourceProgress
   :members:

.. autoclass:: batcher.SinkProgress
   :members:
```

## Listeners

A listener receives every query's start, micro-batch, and termination as it happens.
Register one when polling `recent_progress` would miss a batch or arrive too late to act
on it.

```{eval-rst}
.. autoclass:: batcher.StreamingQueryListener
   :members:

.. autoclass:: batcher.QueryStartedEvent
   :members:

.. autoclass:: batcher.QueryProgressEvent
   :members:

.. autoclass:: batcher.QueryTerminatedEvent
   :members:

.. autofunction:: batcher.add_streaming_listener

.. autofunction:: batcher.remove_streaming_listener

.. autofunction:: batcher.streaming_listeners
```

## Sinks

`ForeachWriter` is the open / process / close shape {py:meth}`ds.write.for_each <batcher.api.io_namespace.writer.Writer.for_each>` accepts for a
destination that needs a connection.

```{eval-rst}
.. autoclass:: batcher.ForeachWriter
   :members:
```

## See also

- {doc}`/user-guide/moving-data/streaming/index`: sources, sinks, triggers, and checkpoints.
- {doc}`/user-guide/moving-data/streaming/monitoring`: reading these records in practice.
- {doc}`/api/operations/configuration`: the `streaming` config section.
