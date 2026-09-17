# Join estimates

`ds.meta.against(other)` asks the question worth asking before a join: will it produce anything at all? A join that returns zero rows because the keys never overlap is one of the quietest failures a pipeline has, and the key ranges can often say so before you pay for the join.

The script checks key overlap and estimated join size on a matching pair of tables, catches a pair whose ids never overlap, and wraps that in a guard that raises before the join runs. The second half reads `ds.meta.approx`, the sketch-backed statistics, which return `None` rather than guess when nothing has been recorded for a column.

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
