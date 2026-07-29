# Asking about a join before running it, and reading approximate statistics

``ds.meta.against(other)`` answers the question that saves the most time in practice: will this join produce anything at all? A join that silently returns zero rows because the keys never overlap is one of the most common quiet failures in a pipeline.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/meta_comparison.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_comparison.py
```
