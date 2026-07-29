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
