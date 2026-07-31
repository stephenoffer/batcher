# Cleaning and subsetting

Deciding which rows survive, and proving the decision was reproducible.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

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
