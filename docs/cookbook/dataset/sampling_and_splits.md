# Sampling and splitting: reproducible subsets that do not leak

Every one of these takes a seed, because an unseeded split is a split you cannot reproduce when the result looks wrong. ``stratified_split`` preserves class balance; a plain random split does not, and on an imbalanced problem that matters.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/sampling_and_splits.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/sampling_and_splits.py
```
