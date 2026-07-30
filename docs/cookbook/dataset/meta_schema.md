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

## See also

- {doc}`meta_predicates`: cheap yes/no questions about the data, and the column-check shorthands.
- {doc}`null_handling`: dataset-level null handling: dropping, filling, and counting missing values.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
