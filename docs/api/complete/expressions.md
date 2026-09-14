# Expressions and selectors

The column language: the functions that build an expression, the selectors that stand for
a set of columns, and the typed accessor namespaces an expression carries.

## Expression constructors

The free functions that produce an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`. A
column reference, a literal, a conditional, and the constructors for the composite types.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   col
   lit
   when
   coalesce
   nullif
   iff
   element
   struct
   named_struct
   map_from_arrays
   array
```

## Column selectors

A *selector* stands for every column matching a predicate. Pass one anywhere a column is expected, such as {py:meth}`ds.select(bt.numeric()) <batcher.Dataset.select>` or {py:meth}`ds.with_columns(bt.floating().round(2)) <batcher.Dataset.with_columns>`, and it expands against the input schema. See the {doc}`transformations guide </user-guide/transform/rows/transformations>` for how they compose.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   all
   numeric
   integer
   floating
   string
   boolean
   temporal
   by_dtype
   matches
   starts_with
   ends_with
   contains
   exclude
```

`Selector` subclasses `Expr`, and everything else follows from that. The scalar algebra
composes onto a selector exactly as it composes onto a column, and applies to every column
the selector matched, so `bt.floating().round(2)` is one expression that rounds however
many floating columns the input turns out to have. Two selectors combine with `|`, `&` and `-`, and `~` takes the complement. The
`.name` namespace renames the outputs.

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.selectors.Selector
   :members:

.. autoclass:: batcher.plan.expr_ir.selectors.core._SelectorNameNamespace
   :members:
   :member-order: bysource
```

## Expr and AggExpr

`Expr` is the fluent builder every column operation returns. Call an aggregate or a window
function on one and you get an `AggExpr` back, which adds `.over(...)` for binding a
window. Nothing here evaluates; an expression is a tree the plan carries into Rust.

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.core.Expr
   :members:
   :member-order: groupwise

.. autoclass:: batcher.AggExpr
   :members:
   :member-order: groupwise
```

### Expression accessors

The typed namespaces reached as `col("x").str`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>`, and {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>`. Multimodal columns add {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>`; biological sequence columns add {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>`.

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.namespaces.strings._StrNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.temporal._DtNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._ListNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._StructNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._JsonNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._MapNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.image._ImageNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.audio._AudioNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.video._VideoNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.sequence._SeqNamespace
   :members:
   :member-order: bysource
```

## See also

- {doc}`/api/relational/expressions`: the same surface with runnable examples, grouped by the job each method does.
- {doc}`/api/relational/expression-accessors`: every accessor method enumerated in tables, which is faster to scan than the generated listing above.
- {doc}`/api/relational/functions`: the scalar, horizontal, aggregate, and window functions these constructors sit beside.
- {doc}`/user-guide/transform/columns/expressions`: the mental model behind an expression, and when it evaluates.
