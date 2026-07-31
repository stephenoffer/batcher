# Expression cookbook

This section holds 34 runnable recipes for the expression API, grouped by the column type they operate on.

Every one runs in Rust over whole columns rather than row by row. Each page embeds a complete, self-contained script from the [`examples/expressions/`](https://github.com/batcher/batcher/tree/main/examples/expressions) directory, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/expressions/scalar/index` | 8 | Arithmetic, branching, nulls and casts, selectors, aggregates, and windows |
| {doc}`/cookbook/expressions/strings/index` | 14 | The `.str` accessor, in three groups |
| {doc}`/cookbook/expressions/temporal/index` | 5 | The `.dt` accessor: parts, arithmetic, truncation, and zones |
| {doc}`/cookbook/expressions/nested/index` | 7 | Lists, structs, maps, and JSON held in a column |

## See also

- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/user-guide/transform/expression-accessors`: the guide to `.str`, `.dt`, `.list`, `.struct`, and `.json`.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
- {doc}`/cookbook/dataset/index`: the same recipe treatment for the `Dataset` verbs.

```{toctree}
:hidden:

scalar/index
strings/index
temporal/index
nested/index
```
