# Joins

The join type decides what happens to rows with no match, which is where most join bugs live. An inner join silently drops them; a left join keeps them with nulls. Decide which you meant before you write it, then assert the row count.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/joins.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/joins.py
```

## See also

- {doc}`/cookbook/dataset/verbs/iteration`: batches, rows, slices, and the single-value cases.
- {doc}`/cookbook/dataset/inspecting/meta_columns`: bounds, uniqueness, nulls, and constancy.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
