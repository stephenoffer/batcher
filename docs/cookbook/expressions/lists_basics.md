# List columns: indexing, slicing, joining, and flattening

A list column holds a variable-length array per row. Indexing and slicing stay columnar, so ``.list.get(0)`` over a million rows is one operator rather than a million Python subscripts. ``explode`` is the escape hatch when you want one row per element instead.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_basics.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_basics.py
```

## See also

- {doc}`lists_aggregate`: reducing a list column to one value per row.
- {doc}`lists_set_operations`: treating two list columns as sets: union, intersection, difference, overlap.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
