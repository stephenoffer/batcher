# Schema without execution

`ds.meta.schema` never executes. A plan knows its own output types, so every question here is a field read, and you can branch on whether a column is numeric before deciding what pipeline to build.

The script asks a six-column table about presence and position, runs one type predicate per family, lists the columns of each family, and narrows the dataset to its numeric columns with `select("numeric")`. Row counts are a different matter: `ds.meta.shape()` is free only when the row count already is, and otherwise it counts.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/meta_schema.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_schema.py
```

## See also

- {doc}`/cookbook/dataset/inspecting/meta_predicates`: cheap yes/no questions about the data, and the column-check shorthands.
- {doc}`/cookbook/dataset/cleaning/null_handling`: dropping, filling, and counting missing values.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
