# JSON columns

Semi-structured payloads often arrive as JSON text in a string column. The `.json` accessor runs a path query in the engine and returns a typed column, so you can filter and aggregate on a nested field without a `json.loads` per row.

The script extracts typed values by JSONPath with `extract_int`, `extract_float`, `extract_string`, and `extract_bool`, asks shape questions with `keys`, `array_length`, and `type_of`, tests presence with `exists`, and reads a raw scalar with `value`. It filters and aggregates on a nested field with no Python parsing at all, reads a key that holds a dot by quoting it, decodes a whole document into a typed struct with `decode`, writes a struct back out as JSON with `encode`, and edits a document with `merge_patch`.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/json_columns.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/json_columns.py
```

## Supported JSONPath

Every method that takes a path accepts the same subset of RFC 9535: the *singular* queries, which select at most one value. Any other selector raises a `PlanError` naming it when the expression is built, rather than being skipped. A path held in a column is checked the same way, by the engine, on the first batch.

The table lists each form and what it reads, with the forms that are refused last:

| Path | Reads |
| --- | --- |
| `$` | The whole document. The leading `$` is optional, so `a.b` means `$.a.b`. |
| `$.name` | The member `name`. Any characters but `.`, `[` and `]` may appear in it. |
| `$."x.y"`, `$['x.y']`, `$["x.y"]` | The member `x.y`. Quoting is the only way to reach a key that holds a dot. A backslash escapes a quote inside the name. |
| `$.a[0]`, `$.a[-1]` | An array element, counted from the front, or from the back for a negative index. Out of range is null. |
| `$.a[*]` | Refused by the `.json` methods: use `.json.values("$.a")`. SQL's `json_extract(j, '$.a[*]')` and `json_extract_string(j, '$.a[*]')` accept it as the last step and return a list, as DuckDB does. |
| `$..a`, `$.a[0:2]`, `$.a[0,1]`, `$.a[?(@.x)]`, `$.a.*` | Refused: recursive descent, slices, unions, filters and wildcards select several values. |
| `$.a[#-1]` | Refused, with a hint. It is DuckDB's last-element spelling. Write `$.a[-1]`. |

`decode` reads a whole document against a declared type instead of one path at a time. Its parse policy follows DuckDB's `json_transform`: a missing key, a JSON `null`, a value of the wrong shape and an unparseable document all decode to null, and `strict=True` raises on each of them instead, as `json_transform_strict` does. A decoded null can't tell a missing key from a JSON `null`, so pair it with `exists` when that difference matters.

## See also

- {doc}`/cookbook/expressions/scalar/horizontal`: reducing across columns instead of down rows.
- {doc}`/cookbook/expressions/nested/lists_aggregate`: reducing a list column to one value per row.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
