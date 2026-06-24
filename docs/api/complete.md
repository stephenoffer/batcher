# Complete reference

Every public name in `batcher`, generated from the source docstrings — the
exhaustive backstop behind the [quick reference](reference.md) and the
example-first [area pages](index.md). Each top-level function links to its own page.

## Top-level functions

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
   greatest
   least
   array
   atan2
   count
   lag
   lead
   first_value
   last_value
   row_number
   rank
   dense_rank
   percent_rank
   cume_dist
   ntile
   range
   date_range
   read
   sql
   catalog
   compact
   engine_version
   from_arrow
   from_batches
   from_pydict
   from_numpy
   from_pandas
   from_polars
   from_spark
   from_dask
   from_huggingface
   from_torch
   from_tf
   set_config
   config_context
```

## Dataset

```{eval-rst}
.. autoclass:: batcher.Dataset
   :members:
   :member-order: groupwise
```

## GroupBy

```{eval-rst}
.. autoclass:: batcher.GroupBy
   :members:
   :member-order: groupwise
```

## Expressions

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.core.Expr
   :members:
   :member-order: groupwise
```

## Configuration

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

.. autoclass:: batcher.PIDConfig
   :members:

.. autoclass:: batcher.MetadataConfig
   :members:
```
