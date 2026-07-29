# Bucketing timestamps: truncate to a period, or snap to a period boundary

``truncate``/``floor`` round a timestamp down to a unit, which is how you build an hourly or daily rollup key. The ``*_start``/``*_end`` pairs snap to calendar boundaries, which is what a month-over-month report needs.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/temporal_truncation.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_truncation.py
```
