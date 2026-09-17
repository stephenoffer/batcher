# Reshaping

Long-to-wide and back is the most common reshape in reporting. Pivot has to know which columns it will produce, so by default it runs an eager pre-pass over `on` to discover them. Pass `columns=[...]` and that pre-pass goes away. Unpivot needs none of it, and it is usually the direction a downstream model wants anyway.

The script pivots a quarterly revenue table wide and unpivots it back, then covers `explode` for list columns, `unnest` for struct columns, the `union`, `intersect`, and `except_` set operations, `with_row_index`, and `crosstab` as a one-call frequency pivot.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/reshaping.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/reshaping.py
```

## See also

- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join for time series.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
