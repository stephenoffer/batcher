# SQL over the same engine, and mixing SQL with DataFrame verbs

``bt.sql`` and ``ds.sql`` build the *same* logical plan the DataFrame API builds, so there is no second engine and no second semantics. That means you can write the join in SQL and the feature engineering in expressions, in one pipeline.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/sql_interface.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/sql_interface.py
```

## See also

- {doc}`sampling_and_splits`: sampling and splitting: reproducible subsets that do not leak.
- {doc}`reshaping`: reshaping: pivot, unpivot, explode, unnest, and set operations.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
