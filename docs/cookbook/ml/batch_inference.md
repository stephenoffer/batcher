# Batch inference: a model over every row, without a Python loop

``map_batches`` hands your callable a whole **pyarrow ``RecordBatch``**, never one row, so ``batch["col"]`` is an Arrow array. Call ``.to_pylist()`` once per batch rather than indexing it element by element. Passing a *class* rather than a function is what makes the model load once per worker instead of once per batch, which on a real model is the difference between minutes and hours.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/batch_inference.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/batch_inference.py
```

## See also

- {doc}`classifiers`: classifiers that fit in the engine: naive Bayes, discriminant analysis, and baselines.
- {doc}`clustering_and_decomposition`: unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD.
- {doc}`../../ml/index`: the ML surface these recipes sit on.
- {doc}`../../ml/preprocessors/index`: the fit and transform steps most pipelines start with.
