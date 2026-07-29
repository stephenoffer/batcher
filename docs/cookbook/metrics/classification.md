# Classification metrics computed as aggregates over a predictions table

These are aggregate expressions, so evaluation is a ``select`` (or a ``group_by`` if you want the metric per segment) rather than a pull into pandas. On a table too big for memory that difference is the whole ballgame.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/classification.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/classification.py
```
