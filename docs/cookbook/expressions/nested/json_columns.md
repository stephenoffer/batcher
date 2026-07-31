# JSON columns

Semi-structured payloads arrive as text. The ``.json`` accessor runs a path query in the engine and returns a typed column, so you can filter and aggregate on a nested field without a ``json.loads`` per row.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/json_columns.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/json_columns.py
```

## See also

- {doc}`/cookbook/expressions/scalar/horizontal`: reducing across columns instead of down rows.
- {doc}`/cookbook/expressions/nested/lists_aggregate`: reducing a list column to one value per row.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
