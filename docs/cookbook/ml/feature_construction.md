# Building new features: interactions, ratios, calendar parts, lags, and rolling windows

These are the featurizers that turn a raw table into a model-ready one. The time-series ones (``LagFeaturizer``, ``RollingFeaturizer``) need an ``order_by`` and usually a ``partition_by``: forgetting the partition silently leaks one entity's history into another's features.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/feature_construction.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/feature_construction.py
```
