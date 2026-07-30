# Data-quality contracts: validate, fail, drop, or quarantine

The four terminal calls are the whole design. ``validate()`` reports without changing the data, ``fail()`` raises, ``drop()`` silently removes bad rows, and ``quarantine()`` splits them out so you can inspect them. Choosing between them is a decision about who is responsible for the bad rows.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/dq_contracts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/dq_contracts.py
```

## See also

- {doc}`deduplication`: deduplication: exact keys, whole rows, and keeping a chosen survivor.
- {doc}`grouping`: grouping: agg, multi-key rollups, and the cube/rollup/grouping-set variants.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
