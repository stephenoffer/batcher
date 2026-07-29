# A/B test statistics computed in the engine: effect size, t-statistic, and intervals

The whole test is aggregate expressions over the assignment table, so it runs where the data is instead of pulling a sample into SciPy. ``welch_*`` does not assume equal variances, which is the right default for a real experiment.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/ab_test_inference.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/ab_test_inference.py
```
