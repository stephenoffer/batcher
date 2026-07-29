# Asking about a dataset's shape without executing it

``ds.meta`` is the introspection accessor. Schema questions are answered from the plan, so they cost nothing: you can branch on whether a column is numeric before deciding what pipeline to build, without touching a row.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/meta_schema.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_schema.py
```
