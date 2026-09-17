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

## See also

- {doc}`/cookbook/ml/estimators/classifiers`: naive Bayes, discriminant analysis, and baselines.
- {doc}`/cookbook/ml/preprocessing/feature_construction`: interactions, ratios, calendar parts, lags, and rolling windows.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/evaluation/index`: scoring the model once it is fitted, as engine expressions.
