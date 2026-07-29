# Unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD

Clustering appends a label column; decomposition appends component columns. Both are transformations of the Dataset, so the result composes with everything else -- you can cluster, then group by the cluster, in one chain.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/clustering_and_decomposition.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/clustering_and_decomposition.py
```
