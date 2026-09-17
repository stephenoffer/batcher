# Path-like strings

Object-storage listings arrive as one long URI per row. Splitting them in the engine keeps the partition key, the extension, and the directory available as ordinary columns you can group and filter by.

The script parses paths with `parse_path`, `parse_filename`, and `parse_dirpath`, recovers a Hive-style partition value from the directory name with `extract`, and counts files per extension. It flags two traps along the way: `parse_dirname` is the root segment rather than the immediate parent, and a non-matching `extract` returns an empty string rather than a null.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_paths.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_paths.py
```

## See also

- {doc}`/cookbook/expressions/strings/shaping/strings_padding`: fixed-width keys and cleaning stray whitespace.
- {doc}`/cookbook/expressions/strings/matching/strings_predicates`: the screen in front of an expensive stage.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
