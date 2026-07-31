# Streaming basics

Batcher runs batch and streaming on one operator set, so the transformation you tested on a file is the one that runs on the stream. This example uses the bounded `rate` source so it terminates, but the pipeline shape is the same for Kafka.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/streaming_basics.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/streaming_basics.py
```

## See also

- {doc}`observability`: verbosity, logging, and execution statistics.
- {doc}`memory_and_caching`: caching a reused branch and spilling under a tight budget.
- {doc}`/user-guide/operate/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/observability`: what the engine records about a run, and where.
