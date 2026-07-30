# Batch as the bounded case of streaming: the same operators, incrementally

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

- {doc}`observability`: watching a query run: verbosity, logging, and execution statistics.
- {doc}`memory_and_caching`: bounded memory: caching a reused branch and spilling under a tight budget.
- {doc}`../../user-guide/performance`: measuring and tuning a query that is correct but slow.
- {doc}`../../user-guide/observability`: what the engine records about a run, and where.
