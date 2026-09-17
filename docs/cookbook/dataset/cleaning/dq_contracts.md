# Data-quality contracts

A data-quality contract is a list of checks plus a decision about what happens to the rows that fail them. The four terminal calls are that decision. `validate()` reports without changing the data, `fail()` raises, `drop()` removes bad rows, and `quarantine()` splits them out so you can inspect them. Choosing between them is choosing who is responsible for the bad rows.

The script builds one contract from `not_null`, `unique`, `in_range`, `matches`, `accepted_values`, and a custom `check`, runs it through all four terminal calls against the same five orders, and asserts which rows each keeps. It ends with `foreign_key`, the referential-integrity check against another dataset.

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
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
