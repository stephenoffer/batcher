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

## See also

- {doc}`meta_columns`: profiling one column: bounds, uniqueness, nulls, and constancy.
- {doc}`meta_predicates`: cheap yes/no questions about the data, and the column-check shorthands.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
