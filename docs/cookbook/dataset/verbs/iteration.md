# Getting results out

Prefer ``iter_batches``: it streams and stays columnar. ``iter_rows`` exists for the cases where you genuinely need one row at a time, and it is the slowest way to leave the engine, so treat reaching for it as a signal that the work belonged in an expression.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/iteration.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/iteration.py
```

## See also

- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join for time series.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
