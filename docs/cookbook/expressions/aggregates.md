# The aggregate vocabulary: counts, positions, quantiles, and approximations

Exact aggregates read every row. The ``approx_*`` family reads sketches instead, trading a bounded error for a large constant-factor speedup and, more importantly, bounded memory on a high-cardinality column.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/aggregates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/aggregates.py
```
