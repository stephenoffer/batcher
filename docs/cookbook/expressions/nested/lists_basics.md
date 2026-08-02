# List basics

A list column holds a variable-length array per row. Indexing and slicing stay columnar, so ``.list.get(0)`` over a million rows is one operator rather than a million Python subscripts. ``explode`` is the escape hatch when you want one row per element instead.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/lists_basics.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_basics.py
```

## See also

- {doc}`/cookbook/expressions/nested/lists_aggregate`: reducing a list column to one value per row.
- {doc}`/cookbook/expressions/nested/lists_set_operations`: union, intersection, difference, overlap.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
