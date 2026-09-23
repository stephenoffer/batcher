# Expr: features, introspection, and the IR

This page lists the rest of the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` surface: the feature transforms and activation functions that let a model's preprocessing stay in the plan, the properties that reach each typed accessor, the `.meta` namespace that reads an expression's own tree, and the Python protocols an expression answers to.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.core
```

## Scaling and feature engineering

Column-wide transforms that standardize, scale, normalize, bin, encode, or flag outliers.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.zscore
   Expr.minmax_scale
   Expr.maxabs_scale
   Expr.mean_center
   Expr.normalize_l1
   Expr.pct_of_total
   Expr.cumulative_pct
   Expr.softmax
   Expr.label_encode
   Expr.cut
   Expr.is_outlier
```

## Activation functions

The neural-network activations, each applied element-wise to a numeric column.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.relu
   Expr.leaky_relu
   Expr.elu
   Expr.gelu
   Expr.sigmoid
   Expr.logit
   Expr.softplus
   Expr.softsign
   Expr.silu
   Expr.mish
   Expr.hardsigmoid
   Expr.hardswish
   Expr.hardtanh
   Expr.tanhshrink
```

## Accessor namespaces

The properties that reach each typed namespace, documented in full on the pages linked from the expression accessors section below.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.str
   Expr.dt
   Expr.list
   Expr.struct
   Expr.json
   Expr.map
   Expr.image
   Expr.audio
   Expr.video
   Expr.seq
   Expr.meta
```

## Expression introspection

The `.meta` accessor reads an expression's tree rather than any row: the output name it would take, the columns it reads, and a drawing of the tree the engine receives.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.meta

.. autoclass:: _MetaNamespace
   :no-members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   _MetaNamespace.output_name
   _MetaNamespace.root_names
   _MetaNamespace.is_column
   _MetaNamespace.has_multiple_outputs
   _MetaNamespace.tree_format
```

## Python protocols and the IR

How an expression behaves under Python's indexing, iteration, and membership syntax, and how it serializes to the engine's JSON IR.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.core

.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.__getitem__
   Expr.__contains__
   Expr.__iter__
   Expr.__len__
   Expr.to_ir
```

## See also

- {doc}`/api/accessors/index`: the methods behind each of those accessor properties.
- {doc}`expression-methods`: the per-row arithmetic these transforms build on.
- {doc}`/api/models/preprocessors`: the fitted preprocessors, when a transform needs statistics from the data.
- {doc}`/architecture/deep-dives/query/plan-ir`: the JSON IR `to_ir` produces.
