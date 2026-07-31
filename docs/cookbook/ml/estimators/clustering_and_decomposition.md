# Clustering and decomposition

Clustering appends a label column; decomposition appends component columns. Both are transformations of the Dataset, so the result composes with everything else. You can cluster, then group by the cluster, in one chain.

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
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
