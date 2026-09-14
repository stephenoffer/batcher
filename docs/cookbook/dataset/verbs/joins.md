# Joins

The join type decides what happens to rows with no match, which is where most join bugs live. An inner join drops them without a word. A left join keeps them with nulls. Decide which you meant before you write it, then assert the row count.

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

- {doc}`/cookbook/dataset/inspecting/meta_comparison`: asking about a join before running it, and reading approximate statistics.
- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
