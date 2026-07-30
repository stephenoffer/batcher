# Complete reference

This page lists the public names in `batcher` that don't have a dedicated area page,
rendered from the source docstrings. It's the exhaustive backstop behind the
{doc}`quick reference <reference>` and the example-first {doc}`area pages <index>`. Each
top-level function links to its own page.

Two large surfaces have their own pages rather than sitting here: {doc}`functions` for
the scalar, horizontal, aggregate, and window functions, and {doc}`metrics` for the
scoring and statistical aggregates.

## Construction and I/O

Build a `Dataset` from in-memory data, another framework, or a storage source, and
register SQL functions or sessions.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_pydict
   from_dict
   from_pylist
   from_dicts
   from_records
   from_items
   from_iter
   from_arrow
   from_batches
   from_numpy
   from_pandas
   from_polars
   from_duckdb
   from_spark
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   from_any
   read
   read_table
   read_csv
   read_parquet
   read_json
   read_ndjson
   read_ipc
   read_orc
   read_avro
   read_excel
   read_delta
   read_iceberg
   read_database
   read_memory
   sql
   streams
   await_any_termination
   register_function
   udf
   compact
   vacuum
   engine_version
   versions
   show_versions
   accelerators
   show_accelerators
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

A *selector* stands for every column matching a predicate. Pass one anywhere a column is expected, such as `ds.select(bt.numeric())` or `ds.with_columns(bt.floating().round(2))`, and it expands against the input schema. See the {doc}`transformations guide <../user-guide/transformations>` for how they compose.

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

## Configuration functions

Read and override the engine tunables. See the {doc}`configuration guide <configuration>`.

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

### Options by name

Address any tunable by its dotted path, in the style of `pandas.set_option` and
`spark.conf.set`. See the {doc}`configuration guide <../configuration/index>`.

```{eval-rst}
.. autofunction:: batcher.config.get_option
.. autofunction:: batcher.config.set_option
.. autofunction:: batcher.config.reset_option
.. autofunction:: batcher.config.option_context
.. autofunction:: batcher.config.option_names
.. autofunction:: batcher.config.describe_options
```

### Serialization

```{eval-rst}
.. autofunction:: batcher.config.config_to_dict
.. autofunction:: batcher.config.env_var_names
```

### Logging and verbosity

One-line switches over `ObservabilityConfig`. See
{doc}`observability <../user-guide/observability>`.

```{eval-rst}
.. autofunction:: batcher.config.set_log_level
.. autofunction:: batcher.config.enable_logging
.. autofunction:: batcher.config.disable_logging
.. autofunction:: batcher.config.set_verbosity
.. autofunction:: batcher.config.set_progress
.. autofunction:: batcher.config.get_logger
```

### Metrics export

Process-wide counters as plain data, for Prometheus, OpenTelemetry, or a log line.

```{eval-rst}
.. autofunction:: batcher.observe.metrics_snapshot
.. autofunction:: batcher.observe.prometheus_text
.. autofunction:: batcher.observe.start_metrics
.. autofunction:: batcher.observe.reset_metrics
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

See the {doc}`governance guide <../user-guide/governance>` for how these fit together.

```{eval-rst}
.. autoclass:: batcher.SecurityCatalog
   :members:

.. autoclass:: batcher.Principal
   :members:

.. autoclass:: batcher.GovernanceEvent
   :members:

.. autofunction:: batcher.security

.. autofunction:: batcher.authenticate

.. autofunction:: batcher.set_verifier

.. autofunction:: batcher.current_verifier

.. autofunction:: batcher.cancel_query

.. autofunction:: batcher.running_queries
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

The `ds.meta` namespace and the accessors it hands out read answers from footers, manifests, and catalogs instead of from the data. See the {doc}`metadata shortcuts guide <../user-guide/metadata-shortcuts>`.

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

The tunables, grouped by subsystem. See the {doc}`configuration guide <configuration>`
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

.. autoclass:: batcher.config.AcceleratorConfig
   :members:

.. autoclass:: batcher.config.EnergyConfig
   :members:

.. autoclass:: batcher.config.DeviceHealthConfig
   :members:

.. autoclass:: batcher.GovernanceConfig
   :members:

.. autoclass:: batcher.TenantConfig
   :members:

.. autofunction:: batcher.tenant

.. autoclass:: batcher.config.config.ObservabilityConfig
   :members:

.. autoclass:: batcher.config.config.ShuffleTlsConfig
   :members:
```


## See also

- {doc}`reference`: the same surface as a short lookup table rather than a full listing.
- {doc}`dataset`: the `Dataset` methods, with the semantics behind each one.
- {doc}`expressions`: the `Expr` surface these methods take.
- {doc}`../user-guide/index`: the task-oriented guides behind this reference.
