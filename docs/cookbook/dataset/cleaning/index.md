# Cleaning and subsetting

Deciding which rows survive, and making that decision reproducible. Deduplication and null handling are the two choices every table forces on you. Quality contracts turn the rest of your assumptions into checks that fail, drop, or quarantine, and the sampling recipe makes every random subset repeatable with a seed.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/dataset/cleaning/deduplication` | Exact keys, whole rows, and keeping a chosen survivor |
| {doc}`/cookbook/dataset/cleaning/null_handling` | Dropping, filling, and counting missing values |
| {doc}`/cookbook/dataset/cleaning/dq_contracts` | Validate, then fail, drop, or quarantine |
| {doc}`/cookbook/dataset/cleaning/sampling_and_splits` | Reproducible subsets that do not leak |

```{toctree}
:hidden:

deduplication
null_handling
dq_contracts
sampling_and_splits
```
