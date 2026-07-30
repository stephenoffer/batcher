# Branching inside an expression: when/then/otherwise, and the SQL null helpers

``bt.when(...).then(...).otherwise(...)`` is the columnar ``if``. Chain ``.when()`` for more branches; the first matching branch wins, exactly like SQL ``CASE``. Because it is an expression it runs in Rust, so a five-way bucketing is still one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/conditionals.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/conditionals.py
```

## See also

- {doc}`column_selectors`: selectors: naming columns by type or pattern instead of one at a time.
- {doc}`horizontal`: horizontal functions: reducing across columns instead of down rows.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
