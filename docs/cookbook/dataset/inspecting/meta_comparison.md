# Join estimates

`ds.meta.against(other)` asks the question worth asking first: will this join produce anything at all? A join that returns zero rows because the keys never overlap is one of the quietest failures a pipeline has. The two footers usually knew.

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
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
