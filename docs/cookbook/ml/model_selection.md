# Cross-validation, learning curves, and feature importance -- all in the engine

``cross_val_score`` takes a ``fit`` and a ``predict`` callable, so it works with the built-in estimators or with anything you wrap. Pass ``key=`` when rows share a group that must not straddle a fold: that is the difference between an honest score and a leak.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/model_selection.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/model_selection.py
```
