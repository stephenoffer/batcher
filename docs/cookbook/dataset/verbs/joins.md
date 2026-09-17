# Joins

The join type decides what happens to a row with no match, and that is where most join bugs live. An inner join drops it without a word. A left join keeps it with nulls.

The script runs one orders table against one customers table through six `how=` values: inner, left, right, outer, semi, and anti, the last being the orphan check. It then covers differently named keys, a cross join, and `join_asof` for the "what was the price when this trade happened" question. It ends on the guard worth copying into a real pipeline: count both sides and assert how many rows the join dropped.

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
