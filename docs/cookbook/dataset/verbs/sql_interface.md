# The SQL interface

``bt.sql`` and ``ds.sql`` build the *same* logical plan the DataFrame API builds, so there is no second engine and no second semantics. That means you can write the join in SQL and the feature engineering in expressions, in one pipeline.

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

- {doc}`/cookbook/dataset/cleaning/sampling_and_splits`: reproducible subsets that do not leak.
- {doc}`/cookbook/dataset/verbs/reshaping`: pivot, unpivot, explode, unnest, and set operations.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
