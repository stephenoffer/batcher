# JSON columns

Semi-structured payloads often arrive as JSON text in a string column. The `.json` accessor runs a path query in the engine and returns a typed column, so you can filter and aggregate on a nested field without a `json.loads` per row.

The script extracts typed values by JSONPath with `extract_int`, `extract_float`, `extract_string`, and `extract_bool`, asks shape questions with `keys`, `array_length`, and `type_of`, tests presence with `exists`, and reads a raw scalar with `value`. It ends by filtering and aggregating on a nested field with no Python parsing at all.

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
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
