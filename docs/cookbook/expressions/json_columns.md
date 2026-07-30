# Reading JSON held in a string column, without parsing it in Python

Semi-structured payloads arrive as text. The ``.json`` accessor runs a path query in the engine and returns a typed column, so you can filter and aggregate on a nested field without a ``json.loads`` per row.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/json_columns.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/json_columns.py
```

## See also

- {doc}`horizontal`: horizontal functions: reducing across columns instead of down rows.
- {doc}`lists_aggregate`: reducing a list column to one value per row.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
