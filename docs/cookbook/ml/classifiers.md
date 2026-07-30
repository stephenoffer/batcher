# Classifiers that fit in the engine: naive Bayes, discriminant analysis, and baselines

Reach for a dummy baseline first. A model that cannot beat "always predict the most frequent class" is not a model, and on an imbalanced problem that baseline can look deceptively strong on accuracy alone.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/classifiers.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/classifiers.py
```

## See also

- {doc}`batch_inference`: batch inference: a model over every row, without a Python loop.
- {doc}`clustering_and_decomposition`: unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD.
- {doc}`../../ml/index`: the ML surface these recipes sit on.
- {doc}`../../ml/preprocessors/index`: the fit and transform steps most pipelines start with.
