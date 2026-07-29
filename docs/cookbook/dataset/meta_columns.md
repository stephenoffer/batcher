# Profiling one column: bounds, uniqueness, nulls, and constancy

This is the accessor to reach for before writing a data-quality rule. Rather than guessing a threshold, ask the column what it actually contains, then encode the answer as a check.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/meta_columns.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_columns.py
```
