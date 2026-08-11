# Expression cookbook

This section holds 39 runnable recipes for the expression API, grouped by the column type they operate on.

Every one runs in Rust over whole columns rather than row by row. Each page embeds a complete, self-contained script from the [`examples/expressions/`](https://github.com/batcher/batcher/tree/main/examples/expressions) directory, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/expressions/scalar/index` | 8 | Arithmetic, branching, nulls and casts, selectors, aggregates, and windows |
| {doc}`/cookbook/expressions/strings/index` | 14 | The {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` accessor, in three groups |
| {doc}`/cookbook/expressions/temporal/index` | 5 | The {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>` accessor: parts, arithmetic, truncation, and zones |
| {doc}`/cookbook/expressions/nested/index` | 7 | Lists, structs, maps, and JSON held in a column |
| {doc}`/cookbook/expressions/genomics/index` | 5 | The {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` accessor: nucleotide, protein, and FASTQ-quality columns |

## See also

- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/user-guide/transform/columns/expression-accessors`: the guide to {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>`.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
- {doc}`/cookbook/dataset/index`: the same recipe treatment for the {py:class}`Dataset <batcher.Dataset>` verbs.

```{toctree}
:hidden:

scalar/index
strings/index
temporal/index
nested/index
genomics/index
```
