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

## See also

- {doc}`column_selectors`: selectors: naming columns by type or pattern instead of one at a time.
- {doc}`conditionals`: branching inside an expression: when/then/otherwise, and the SQL null helpers.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
