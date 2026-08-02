# Join estimates

``ds.meta.against(other)`` answers the question that saves the most time in practice: will this join produce anything at all? A join that silently returns zero rows because the keys never overlap is one of the most common quiet failures in a pipeline.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/meta_comparison.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_comparison.py
```

## See also

- {doc}`/cookbook/dataset/inspecting/meta_columns`: bounds, uniqueness, nulls, and constancy.
- {doc}`/cookbook/dataset/inspecting/meta_predicates`: cheap yes/no questions about the data, and the column-check shorthands.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
