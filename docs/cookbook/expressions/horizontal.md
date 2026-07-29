# Horizontal functions: reducing across columns instead of down rows

An aggregate like ``sum()`` collapses a column. The ``*_horizontal`` family collapses a *row* across several columns and leaves the row count alone. That is what you want for a per-row total, a "did any check fail" flag, or a coalesce-style fallback.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/horizontal.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/horizontal.py
```
