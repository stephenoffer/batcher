# Observability

Observability is configuration, not instrumentation you sprinkle through the pipeline. Turn it up for one block, read what happened, and turn it back down. The query code is unchanged either way.

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

- {doc}`memory_and_caching`: caching a reused branch and spilling under a tight budget.
- {doc}`streaming_basics`: the same operators, incrementally.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
