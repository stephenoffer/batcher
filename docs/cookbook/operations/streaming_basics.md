# Streaming basics

Batcher runs batch and streaming on one operator set, so the transformation you tested on a file is the one that runs on the stream. The script builds its input with `bt.from_pydict` so that it terminates, but the pipeline shape is the same for Kafka. Watch the last assertion. The answer collected in one go and the answer summed over `iter_batches` agree, and everything else on this page rests on that.

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
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
