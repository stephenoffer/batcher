# Grouping: agg, multi-key rollups, and the cube/rollup/grouping-set variants

``group_by().agg()`` is the workhorse. ``rollup`` and ``cube`` compute subtotals in the same pass, which is how you build a report with per-region, per-product, and grand-total rows without three separate queries and a union.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/grouping.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/grouping.py
```
