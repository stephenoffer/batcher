# Training data and features

These pipelines produce the table a model trains on. Between them they cover the four ways
that table usually goes wrong: unfitted transforms, a leaky split, duplicated documents, and
features computed after the label they are supposed to predict.

| Pipeline | What it builds |
|---|---|
| {doc}`Feature pipeline <feature-pipeline>` | A model matrix from a raw table, through fitted transforms |
| {doc}`Train/test split <train-test-split>` | A split that does not leak, and stays the same on the next run |
| {doc}`Training-data dedup <training-data-dedup>` | Near-duplicate removal, because exact dedup barely moves a web crawl |
| {doc}`Recommender features <recommender-features>` | A `(user, item, label)` triple with features attached as of the right moment |

## See also

- {doc}`/ml/training/index`: loaders, distributed training, and corpus preparation.
- {doc}`/cookbook/ml/preprocessing/index`: the individual transforms these pipelines chain.

```{toctree}
:hidden:

feature-pipeline
train-test-split
training-data-dedup
recommender-features
```
