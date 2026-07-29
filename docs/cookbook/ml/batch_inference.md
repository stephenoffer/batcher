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
