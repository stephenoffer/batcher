# Complete reference

This section renders every public name in `batcher` that has no dedicated area page,
straight from the source docstrings. It is the exhaustive backstop behind the
{doc}`quick reference </api/reference>` and the example-first {doc}`area pages </api/index>`.

One page would not work. `Dataset` and `Expr` carry a few hundred members between them, and
with the accessor namespaces alongside, a single listing runs to thousands of signatures
that nothing but browser search can reach. So the surface is split along the lines a
reader looks it up on, and each top-level function still gets its own generated page:

| Page | Holds |
| --- | --- |
| {doc}`construction` | Building a `Dataset` from anything, the readers, and the writer namespace |
| {doc}`expressions` | `col`, `lit`, the selectors, `Expr`, and the typed accessor namespaces |
| {doc}`dataset` | `Dataset`, `GroupBy`, and the `ml`, `dq`, `scd`, and `meta` accessors |
| {doc}`configuration` | Option functions, the config dataclasses, and the result cache |
| {doc}`governance` | Policy, principals, query control, and SQL sessions |

Three surfaces sit elsewhere. {doc}`/api/relational/functions` holds the scalar,
horizontal, aggregate and window functions, {doc}`/api/models/metrics` the scoring and
statistical aggregates, and {doc}`/api/operations/streaming` the triggers, output modes,
query progress and listeners.

## See also

- {doc}`/api/reference`: the same surface as a short lookup table rather than a full listing.
- {doc}`/api/relational/dataset`: the `Dataset` methods, with the semantics behind each one.
- {doc}`/api/relational/expressions`: the {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` surface these methods take.
- {doc}`/api/operations/exceptions`: what each of these calls raises, and which builtin it also subclasses.
- {doc}`/user-guide/index`: the task-oriented guides behind this reference.

```{toctree}
:hidden:

construction
expressions
dataset
configuration
governance
```
