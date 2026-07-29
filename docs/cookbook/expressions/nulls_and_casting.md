# Nulls and type casting: the two places a pipeline quietly changes its answer

Null is not zero and not empty string, and every aggregate skips it. Casting is where a schema mismatch between two sources gets resolved, and where an unparseable value becomes a null rather than an error.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/nulls_and_casting.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/nulls_and_casting.py
```
