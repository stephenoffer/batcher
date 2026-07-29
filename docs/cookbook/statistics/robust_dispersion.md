# Robust spread: quantile-based measures that one outlier cannot move

Standard deviation is a poor summary of a long-tailed distribution, which describes most latency and revenue data. These are the quantile-based alternatives: a single extreme row shifts them barely at all, so a dashboard built on them stops flapping.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/robust_dispersion.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/robust_dispersion.py
```
