# Durations between two timestamp columns, and shifting a timestamp

``*_between`` gives a whole-unit difference between two columns, which is how you compute an age, a lead time, or a session length. ``offset_by`` shifts by a duration string, which is how you build a "30 days ago" cutoff without leaving the expression API.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/temporal_differences.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_differences.py
```
