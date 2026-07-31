# Grouping and rollups

``group_by().agg()`` is the workhorse. ``rollup`` and ``cube`` compute subtotals in the same pass, which is how you build a report with per-region, per-product, and grand-total rows without three separate queries and a union.

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

- {doc}`/cookbook/dataset/cleaning/dq_contracts`: validate, fail, drop, or quarantine.
- {doc}`/cookbook/dataset/verbs/iteration`: batches, rows, slices, and the single-value cases.
- {doc}`/user-guide/transform/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
