# Complete reference

This section renders every public name in `batcher` that has no dedicated area page,
straight from the source docstrings. It is the exhaustive backstop behind the
{doc}`quick reference </api/reference>` and the example-first {doc}`area pages </api/index>`.

One page would not work. `Dataset` and `Expr` carry a few hundred members between them, and
with the accessor namespaces alongside, a single listing runs to thousands of signatures
that nothing but browser search can reach. So the surface is split along the lines a
reader looks it up on. Each page lists its names in tables grouped by task, and every
function, class, and method gets its own generated page:

| Page | Holds |
| --- | --- |
| {doc}`construction` | Building a `Dataset` from anything, the readers grouped by source family, and the writers |
| {doc}`expressions` | `col`, `lit`, the selectors, `Expr` and `AggExpr` grouped by task |
| {doc}`string-accessor` | The `.str` namespace |
| {doc}`temporal-and-nested-accessors` | The `.dt`, `.list`, `.struct`, `.json`, and `.map` namespaces |
| {doc}`multimodal-and-sequence-accessors` | The `.image`, `.audio`, `.video`, and `.seq` namespaces |
| {doc}`dataset` | `Dataset` and `GroupBy`, grouped by task |
| {doc}`dataset-accessors` | The `ml`, `dq`, `scd`, and `meta` accessors a dataset hands out |
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
string-accessor
temporal-and-nested-accessors
multimodal-and-sequence-accessors
dataset
dataset-accessors
configuration
governance
```
