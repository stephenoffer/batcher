# Losses that score a probability or a margin rather than a hard label

A classifier that says "0.51" and one that says "0.99" both predict the positive class, but they are not equally right. These losses read the score column, which is what you need to tell a confident model from a lucky one.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/probabilistic_losses.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/probabilistic_losses.py
```
