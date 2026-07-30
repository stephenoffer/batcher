# Selectors: naming columns by type or pattern instead of one at a time

A selector is an ``Expr`` leaf standing for *every* matching column, so "round every float" is one expression that keeps working when a column is added. Spelling out names is how a pipeline silently stops covering a new column. Combine selectors with ``|``, ``&``, ``-``, and ``~``.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/column_selectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/column_selectors.py
```

## See also

- {doc}`aggregates`: the aggregate vocabulary: counts, positions, quantiles, and approximations.
- {doc}`conditionals`: branching inside an expression: when/then/otherwise, and the SQL null helpers.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
