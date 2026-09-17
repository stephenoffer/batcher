# Deduplication

"Remove duplicates" is under-specified. Which copy survives is the whole decision, and leaving it to an arbitrary one is how a pipeline becomes non-deterministic. The latest row by a version or timestamp is almost always what was meant.

The script contrasts whole-row `distinct` with `distinct(subset=...)` on a key subset, then keeps the latest version per id with a `row_number` window, which is the deterministic spelling. Before deciding anything it counts the duplicated keys, and it shows `ds.meta` answering "is this column a key?" without a full aggregate.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/deduplication.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/deduplication.py
```

## See also

- {doc}`/cookbook/dataset/cleaning/dq_contracts`: validate, fail, drop, or quarantine.
- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
