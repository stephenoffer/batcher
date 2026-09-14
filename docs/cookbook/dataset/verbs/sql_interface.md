# The SQL interface

`bt.sql` and `ds.sql` build the *same* logical plan the DataFrame API builds. No second engine, no second semantics. Write the join in SQL and the feature engineering in expressions, in one pipeline.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/sql_interface.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/sql_interface.py
```

## See also

- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join for time series.
- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
