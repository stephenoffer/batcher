# Reducing a list column to one value per row

These are per-row reductions, not group-by aggregates: ``.list.sum()`` sums *within* each row's list and leaves the row count unchanged. That is the difference between "total per basket" and "total across baskets".

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_aggregate.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_aggregate.py
```

## See also

- {doc}`json_columns`: reading JSON held in a string column, without parsing it in Python.
- {doc}`lists_basics`: list columns: indexing, slicing, joining, and flattening.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
