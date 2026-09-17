# Sampling and splits

An unseeded split is one you cannot reproduce when the result looks wrong, so pass `seed` to every call that takes one. `sample_per_group` doesn't take one. It caps each group by an ordering rather than at random, so it is already deterministic.

The script samples a fraction and proves the same seed returns the same rows, splits train, validation, and test on a key so no group leaks across the boundary, and checks that `stratified_split` keeps an 80/20 label balance on both sides. It also covers `sample_per_group`, `balance_classes`, a seeded `shuffle`, and `with_random` for a hand-rolled split.

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
