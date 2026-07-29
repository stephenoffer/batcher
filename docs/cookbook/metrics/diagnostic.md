# Diagnostic metrics: the epidemiology-style view of a binary classifier

Accuracy hides everything on an imbalanced problem. Likelihood ratios, informedness, and markedness describe how much a prediction actually moves your belief, which is the number you want when positives are rare.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/diagnostic.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/diagnostic.py
```
