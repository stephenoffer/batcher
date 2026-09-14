# Getting results out

Prefer `iter_batches`. It streams and stays columnar, so a result far larger than memory still comes back in bounded memory. `iter_rows` streams too, and per-row Python at the *end* of a pipeline is fine. Inside the query it is not, and that work belongs in an expression or a `map_batches`. `to_pylist` materializes the whole result, so reach for it only when you already know the size.

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
