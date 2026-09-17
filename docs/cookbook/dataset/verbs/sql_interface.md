# The SQL interface

{py:func}`bt.sql <batcher.sql>` and `ds.sql` build the *same* logical plan the DataFrame API builds. There is no second engine and no second semantics, so you can write the join in SQL and the feature engineering in expressions, in one pipeline.

The script queries a dataset through `ds.sql`, where the dataset is the table `self`, and asserts that the DataFrame spelling returns an identical result. It registers two tables on a {py:class}`Session <batcher.Session>` to join them by name, chains `with_columns` onto a SQL result, and shows that SQL stays lazy until a terminal call.

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
