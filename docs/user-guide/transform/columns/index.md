# The column language

These pages cover {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, the language every column computation is written in. An expression
lowers to Rust and runs over whole Arrow batches, which is why column work never becomes a
Python loop.

Read {doc}`Expressions <expressions>` first. The rest assume it.

| Page | What it covers |
|---|---|
| {doc}`Expressions <expressions>` | Building, combining, and reusing an `Expr`, and why it replaces a callback |
| {doc}`Expression accessors <expression-accessors>` | The {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` namespaces, one per column kind |
| {doc}`The sequence accessor <sequence-accessor>` | The {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>` namespace: DNA, RNA, protein, and FASTQ-quality columns |
| {doc}`Expression recipes <expression-recipes>` | The jobs people actually reach for the language to do, assembled |
| {doc}`The type system <type-system>` | What each type means here, and why a narrow integer widens at the boundary |
| {doc}`User-defined functions <udfs>` | When an expression genuinely cannot express it, and how to write the batch callback that can |

## See also

- {doc}`/api/relational/expressions`: every `Expr` method, enumerated.
- {doc}`/cookbook/expressions/index`: 39 runnable pages of the same language.

```{toctree}
:hidden:

expressions
expression-accessors
string-accessor
sequence-accessor
expression-recipes
type-system
udfs
```
