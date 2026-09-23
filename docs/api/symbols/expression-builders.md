# Expression constructors and selectors

This page lists the free functions that produce an expression and the selectors that stand for a set of columns. Nothing here reads a row: each one builds a node in the tree the plan carries into Rust.

A selector is the part worth reading before you need it. {py:obj}`bt.numeric() <batcher.numeric>` is not a list of column names resolved when you write it, it is an expression that expands against whatever schema it meets, so `bt.floating().round(2)` is one expression that rounds however many floating columns the input turns out to have.

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

## See also

- {doc}`expression-methods`: the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` these functions return.
- {doc}`/api/relational/expressions`: the same surface with a runnable example per group.
- {doc}`/api/relational/functions`: the scalar, horizontal, aggregate, and window functions these sit beside.
- {doc}`/user-guide/transform/columns/expressions`: the mental model, and when an expression evaluates.
