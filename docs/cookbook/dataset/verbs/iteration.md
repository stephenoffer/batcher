# Getting results out

Every pipeline ends by handing results to something that isn't the engine, and how you do that decides whether the result has to fit in memory. Prefer `iter_batches`. It streams Arrow batches, so a result far larger than memory comes back in bounded memory. `iter_rows` streams too, and per-row Python at the *end* of a pipeline is fine. Inside the query it is not: that work belongs in an expression or a `map_batches`.

The script walks the exits from cheapest to most expensive: batches and slices, row tuples, `limit`/`tail` for a peek, `first`, `last` and `item` for single values, `top_k` instead of a full sort, and `to_pylist` for a result you already know is small. It closes by showing that a Python loop over `iter_rows` and a native `sum` agree, and only one of them stays in Rust.

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
