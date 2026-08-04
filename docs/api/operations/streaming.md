# Streaming API

This page is the reference for Batcher's streaming surface: the trigger and output-mode
values a streaming write takes, what a running query reports, and the listener interface
that receives those reports as they happen.

For how to use them, see {doc}`/user-guide/moving-data/streaming` and
{doc}`/user-guide/moving-data/streaming-monitoring`. The handle a streaming write returns
is documented with the rest of the {py:class}`Dataset <batcher.Dataset>` surface in {doc}`/api/complete`.

## Triggers and output modes

```{eval-rst}
.. autoclass:: batcher.Trigger
   :members:

.. autoclass:: batcher.OutputMode
   :members:
```

### Query progress and status

What a running query reports: one record per completed micro-batch, plus the state each
stateful operator is holding and what it dropped as late.

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

### Listeners

Register a listener to receive every query's start, micro-batch, and termination as it
happens, rather than polling `recent_progress`.

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

- {doc}`/user-guide/moving-data/streaming`: sources, sinks, triggers, and checkpoints.
- {doc}`/user-guide/moving-data/streaming-monitoring`: reading these records in practice.
- {doc}`/api/operations/configuration`: the `streaming` config section.
