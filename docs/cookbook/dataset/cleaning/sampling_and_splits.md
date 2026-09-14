# Sampling and splits

An unseeded split is one you cannot reproduce when the result looks wrong. Pass `seed` to every call that takes one. `sample_per_group` does not: it caps each group by an ordering rather than at random, so it is already deterministic. `stratified_split` preserves class balance where a plain random split does not, and on an imbalanced problem that matters.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/sampling_and_splits.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/sampling_and_splits.py
```

## See also

- {doc}`/cookbook/dataset/cleaning/deduplication`: exact keys, whole rows, and keeping a chosen survivor.
- {doc}`/cookbook/dataset/cleaning/dq_contracts`: validate, fail, drop, or quarantine.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
