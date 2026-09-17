# Grouping and rollups

`group_by().agg()` is the workhorse, and most reports need more than one level of it. `rollup` and `cube` add the subtotal rows, so a report with per-region, per-product and grand-total lines is one call rather than three queries you stack by hand. Underneath they are that stack: one ordinary `group_by` per level, unioned over a shared source list, so the levels share one read of the input.

The script computes several aggregates in one pass, groups by several keys and by a derived key, then compares `rollup`, `cube`, and `grouping_sets` on the same table. It finishes with a filter on the aggregated result, the DataFrame spelling of `HAVING`.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/grouping.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/grouping.py
```

## See also

- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join for time series.
- {doc}`/cookbook/dataset/verbs/iteration`: batches, rows, slices, and the single-value cases.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
