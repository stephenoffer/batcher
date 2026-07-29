# Is this column symmetric, skewed, or heavy-tailed?

Shape decides which summary is honest. On a skewed column the mean is not the typical value, and a normality-assuming test is not valid. These aggregates answer that question before you pick the summary rather than after someone questions the dashboard.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/distribution_shape.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/distribution_shape.py
```
