# Clustering and decomposition

Clustering appends a label column and decomposition appends component columns. Both results are ordinary Datasets, so you can cluster and then group by the cluster in one chain.

The script fits `KMeans` and `GaussianMixture` on two point clouds and aggregates per cluster. It projects the same points with `PCA`, whose components are named `pc1`, `pc2`, and so on, and with `TruncatedSVD`, the same idea without mean-centering. Pass `keep_original=True` when you still need the source columns.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/clustering_and_decomposition.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/clustering_and_decomposition.py
```


## Fitting and labelling in one pass

{py:meth}`KMeans.fit_predict <batcher.ml.KMeans>` fits the model and returns the training
data with its cluster assignment attached. It exists because fitting and then transforming
the same dataset reads it twice, and clustering is the case where the labels you want are
for the very rows you fitted on.

```python
import batcher as bt
from batcher.ml import KMeans

ds = bt.from_pydict(
    {"x": [1.0, 1.2, 8.0, 8.4, 1.1, 8.2], "y": [1.0, 0.9, 8.1, 7.9, 1.2, 8.3]}
)
labelled = KMeans(["x", "y"], n_clusters=2, seed=0).fit_predict(ds)
print(sorted(labelled.to_pydict()["cluster"]))
```

Cluster numbers are arbitrary labels, not an ordering, so compare partitions rather than
individual ids across runs. `seed` makes a run reproducible; it does not make cluster `0`
mean the same thing as `n_clusters` changes.

## See also

- {doc}`/cookbook/ml/estimators/classifiers`: naive Bayes, discriminant analysis, and baselines.
- {doc}`/cookbook/ml/preprocessing/feature_construction`: interactions, ratios, calendar parts, lags, and rolling windows.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/evaluation/index`: scoring the model once it is fitted, as engine expressions.
