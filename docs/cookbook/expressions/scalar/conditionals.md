# Conditionals

``bt.when(...).then(...).otherwise(...)`` is the columnar ``if``. Chain ``.when()`` for more branches; the first matching branch wins, exactly like SQL ``CASE``. Because it is an expression it runs in Rust, so a five-way bucketing is still one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/conditionals.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/conditionals.py
```

## See also

- {doc}`/cookbook/expressions/scalar/column_selectors`: naming columns by type or pattern instead of one at a time.
- {doc}`/cookbook/expressions/scalar/horizontal`: reducing across columns instead of down rows.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
