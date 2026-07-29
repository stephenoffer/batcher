# Scaling numeric features, and why the choice of scaler matters

Every scaler follows the ``fit`` / ``transform`` split for a reason: the statistics come from the training set and are then *applied* to validation and production data. Fitting on everything is the classic leak, and the API makes the correct thing the easy thing.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/preprocessing_scaling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_scaling.py
```
