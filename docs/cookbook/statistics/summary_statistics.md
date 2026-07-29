# Summary aggregates beyond mean and stddev

The averages here answer different questions. Geometric mean is the right average for growth rates, harmonic mean for rates and speeds, RMS for magnitudes that cancel. Using the arithmetic mean for all three is the most common quiet mistake in a metrics table.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/summary_statistics.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/summary_statistics.py
```
