# Watching a query run: verbosity, logging, and execution statistics

Observability is configuration, not instrumentation you sprinkle through the pipeline. Turn it up for one block, read what happened, and turn it back down -- the query code is unchanged either way.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/observability.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/observability.py
```

## See also

- {doc}`memory_and_caching`: bounded memory: caching a reused branch and spilling under a tight budget.
- {doc}`streaming_basics`: batch as the bounded case of streaming: the same operators, incrementally.
- {doc}`../../user-guide/performance`: measuring and tuning a query that is correct but slow.
- {doc}`../../user-guide/observability`: what the engine records about a run, and where.
