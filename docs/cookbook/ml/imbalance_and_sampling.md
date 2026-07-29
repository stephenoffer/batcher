# Class imbalance: measure it, then resample or reweight

Resampling changes the data; weighting changes the loss. Prefer weights when the model supports them, because oversampling duplicates rows (and any leakage in them) while undersampling throws information away.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/imbalance_and_sampling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/imbalance_and_sampling.py
```
