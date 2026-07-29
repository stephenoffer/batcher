# Discretizing, clipping, and reshaping the distribution of a numeric column

Binning turns a continuous variable into a categorical one, which is how you let a linear model express a non-monotonic effect. Clipping and power transforms attack the other problem: a long tail that dominates the loss.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/preprocessing_binning.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/preprocessing_binning.py
```
