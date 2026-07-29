# Quantiles, histograms, and the exact-versus-approximate trade

Exact quantiles need the whole column ordered. Sketch-backed ones need bounded memory and answer within a known error, which is what makes them usable on a column that does not fit in memory. Know which one you are getting.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/statistics/quantiles_and_histograms.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/statistics/quantiles_and_histograms.py
```
