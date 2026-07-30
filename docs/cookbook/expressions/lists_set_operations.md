# Treating two list columns as sets: union, intersection, difference, overlap

This is the shape behind "which tags do these two documents share" and "what did the user add to the cart since last time". Everything is per row and columnar, so a set operation over a million rows never builds a million Python sets.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_set_operations.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_set_operations.py
```

## See also

- {doc}`lists_basics`: list columns: indexing, slicing, joining, and flattening.
- {doc}`lists_transforms`: transforming inside a list column, without exploding it first.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
