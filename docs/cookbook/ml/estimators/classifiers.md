# In-engine classifiers

Reach for a dummy baseline first. A model that cannot beat "always predict the most frequent class" is not a model, and on an imbalanced problem that baseline can look deceptively strong on accuracy alone.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/classifiers.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/classifiers.py
```

## See also

- {doc}`/cookbook/ml/inference/batch_inference`: a model over every row, without a Python loop.
- {doc}`/cookbook/ml/estimators/clustering_and_decomposition`: KMeans, Gaussian mixtures, PCA, and truncated SVD.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
