# Profiling a table you have just been handed

The first thing to do with unfamiliar data is measure it, not query it. These are the one-liners that answer "what is in here" before you write a single business rule.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/profiling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/profiling.py
```
