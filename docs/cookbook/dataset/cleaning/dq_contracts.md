# Data-quality contracts

The four terminal calls are the whole design. ``validate()`` reports without changing the data, ``fail()`` raises, ``drop()`` silently removes bad rows, and ``quarantine()`` splits them out so you can inspect them. Choosing between them is a decision about who is responsible for the bad rows.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/dq_contracts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/dq_contracts.py
```

## See also

- {doc}`/cookbook/dataset/cleaning/deduplication`: exact keys, whole rows, and keeping a chosen survivor.
- {doc}`/cookbook/dataset/verbs/grouping`: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`/user-guide/transform/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
