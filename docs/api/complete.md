# Complete API reference

Every public name in `batcher`, generated from the source docstrings. The other
pages in this section are example-first guides; this one is the exhaustive
reference. Each top-level function links to its own page.

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
   struct
   named_struct
   iff
   ifnull
   nanvl
   concat
   concat_ws
   format_string
   log
   atan2
   gcd
   lcm
   hypot
   width_bucket
   count
   count_if
   corr
   covar_pop
   covar_samp
   lag
   lead
   first_value
   last_value
   nth_value
   row_number
   rank
   dense_rank
   percent_rank
   cume_dist
   ntile
   current_date
   current_timestamp
   date_add
   date_sub
   date_part
   now
   today
   date_range
   range
   read
   sql
   compact
   engine_version
   from_arrow
   from_batches
   from_pydict
   from_pylist
   from_items
   from_numpy
   from_pandas
   from_polars
   from_spark
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   set_config
   config_context
   udf
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
