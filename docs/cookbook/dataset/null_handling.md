# Dataset-level null handling: dropping, filling, and counting missing values

Null propagation is where a pipeline changes its answer without telling you. Decide per column whether a missing value means "unknown" (leave it), "zero" (fill it), or "this row is unusable" (drop it) -- and never let the default decide for you.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/null_handling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/null_handling.py
```
