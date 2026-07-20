# Complete reference

This page lists every public name in `batcher`, rendered from the source docstrings. It's the exhaustive backstop behind the [quick reference](reference.md) and the example-first [area pages](index.md). Each top-level function links to its own page.

## Construction and I/O

Build a `Dataset` from in-memory data, another framework, or a storage source, and
register SQL functions or sessions.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_pydict
   from_pylist
   from_items
   from_arrow
   from_batches
   from_numpy
   from_pandas
   from_polars
   from_spark
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   read
   read_memory
   sql
   streams
   await_any_termination
   register_function
   udf
   compact
   vacuum
   engine_version
   start_ui
   stop_ui
   ui_url
```

## Expressions and columns

Reference and derive columns, build literals, and branch.

```{eval-rst}
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
   array
```

## Column selectors

A *selector* stands for every column matching a predicate. Pass one anywhere a column is expected, such as `ds.select(bt.numeric())` or `ds.with_columns(bt.floating().round(2))`, and it expands against the input schema. See the [transformations guide](../user-guide/transformations.md) for how they compose.

```{eval-rst}
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

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.selectors.Selector
   :members:

.. autoclass:: batcher.plan.expr_ir.selectors.core._SelectorNameNamespace
   :members:
   :member-order: bysource
```

## Scalar functions

Row-wise math, string, and date/time helpers usable anywhere an expression is.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   greatest
   least
   atan2
   arctan2
   hypot
   gcd
   lcm
   log
   nanvl
   width_bucket
   concat
   concat_ws
   format_string
   mask
   hmac_sha256
   aes_encrypt
   aes_decrypt
   current_date
   current_timestamp
   date_add
   date_sub
   date_part
   range
   date_range
   sequence
   hash_rows
```

## Horizontal functions

These reduce *across* the listed expressions within one row, rather than down a column. The vertical counterparts are the aggregates below.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   min_horizontal
   max_horizontal
   sum_horizontal
   mean_horizontal
   count_horizontal
   product_horizontal
   all_horizontal
   any_horizontal
   reduce_horizontal
   fold_horizontal
```

## Aggregate and window functions

Use these in `group_by(...).agg(...)` or `.over(...)` window frames. The ranking and
value functions are window-only: bind them with `.over(partition_by=…, order_by=…)`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   count
   sum
   mean
   min
   max
   median
   std
   var
   n_unique
   product
   mode
   skewness
   kurtosis
   bool_and
   bool_or
   bit_and
   bit_or
   bit_xor
   array_agg
   quantile
   approx_quantile
   approx_median
   approx_n_unique
   histogram
   count_if
   corr
   covar_pop
   covar_samp
   regr_slope
   regr_intercept
   regr_r2
   regr_count
   regr_avgx
   regr_avgy
   regr_sxx
   regr_syy
   regr_sxy
   var_pop
   stddev_pop
   geometric_mean
   harmonic_mean
   rms
   cv
   sem
   midrange
   weighted_mean
   q1
   q3
   iqr
   value_range
   null_rate
   non_null_rate
   nunique_ratio
   first
   last
   arg_min
   arg_max
   row_number
   rank
   dense_rank
   percent_rank
   cume_dist
   ntile
   lag
   lead
   first_value
   last_value
   nth_value
```

## Configuration functions

Read and override the engine tunables. See the [configuration guide](configuration.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   set_config
   config_context
```

```{eval-rst}
.. autofunction:: batcher.config.active_config
```

## Dataset

```{eval-rst}
.. autoclass:: batcher.Dataset
   :members:
   :member-order: groupwise
   :special-members: __getitem__, __len__, __iter__, __contains__, __add__, __or__, __and__, __sub__, __arrow_c_stream__
```

## GroupBy

```{eval-rst}
.. autoclass:: batcher.GroupBy
   :members:
   :member-order: groupwise
```

## Governance

See the [governance guide](../user-guide/governance.md) for how these fit together.

```{eval-rst}
.. autoclass:: batcher.SecurityCatalog
   :members:

.. autoclass:: batcher.Principal
   :members:

.. autoclass:: batcher.GovernanceEvent
   :members:

.. autofunction:: batcher.security
```

Column-level lineage is reached from the dataset itself: :meth:`batcher.Dataset.lineage`.

## Expressions

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.core.Expr
   :members:
   :member-order: groupwise

.. autoclass:: batcher.AggExpr
   :members:
   :member-order: groupwise
```

### Expression accessors

These are the typed namespaces reached as `col("x").str`, `.dt`, `.list`, `.struct`, `.json`, and `.map`. Multimodal columns add `.image`, `.audio`, and `.video`.

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
```

## Reading and writing

`bt.read` is the reader namespace; `ds.write` is the writer namespace.

```{eval-rst}
.. autoclass:: batcher.api.io_namespace.reader.Reader
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.io_namespace.writer.Writer
   :members:
   :member-order: bysource
```

## Dataset accessors

The `ds.ml`, `ds.dq`, and `ds.scd` namespaces for machine learning, data quality,
and slowly-changing-dimension workflows.

```{eval-rst}
.. autoclass:: batcher.api.dataset.ml.DatasetML
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.dq.DatasetDQ
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.dq.ValidationReport
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.scd.DatasetSCD
   :members:
   :member-order: bysource
```

## Metadata shortcuts

The `ds.meta` namespace and the accessors it hands out read answers from footers, manifests, and catalogs instead of from the data. See the [metadata shortcuts guide](../user-guide/metadata-shortcuts.md).

```{eval-rst}
.. autoclass:: batcher.api.dataset.meta.frame.DatasetMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.column.ColumnMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.checks.ColumnChecks
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.schema.SchemaMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.nulls.NullsMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.approx.ApproxMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.storage.StorageMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.pair.PairMeta
   :members:
   :member-order: bysource
```

## SQL sessions

```{eval-rst}
.. autoclass:: batcher.Session
   :members:
   :member-order: groupwise
```

## Streaming

```{eval-rst}
.. autoclass:: batcher.Trigger
   :members:

.. autoclass:: batcher.OutputMode
   :members:
```

## Configuration classes

The tunables, grouped by subsystem. See the [configuration guide](configuration.md)
for what each one does and when to change it.

```{eval-rst}
.. autoclass:: batcher.Config
   :members:

.. autoclass:: batcher.ExecutionConfig
   :members:

.. autoclass:: batcher.MemoryConfig
   :members:

.. autoclass:: batcher.FlowControlConfig
   :members:

.. autoclass:: batcher.OptimizerConfig
   :members:

.. autoclass:: batcher.config.config.CardinalityConfig
   :members:

.. autoclass:: batcher.config.config.CostWeights
   :members:

.. autoclass:: batcher.config.config.CostCoefficients
   :members:

.. autoclass:: batcher.config.config.DistributedConfig
   :members:

.. autoclass:: batcher.PIDConfig
   :members:

.. autoclass:: batcher.MetadataConfig
   :members:

.. autoclass:: batcher.config.config.ObservabilityConfig
   :members:

.. autoclass:: batcher.config.config.ShuffleTlsConfig
   :members:
```
