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

## See also

- {doc}`lists_vectors`: embedding vectors as list columns: similarity, distance, and normalization.
- {doc}`numeric_math`: arithmetic and math functions on numeric columns.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
