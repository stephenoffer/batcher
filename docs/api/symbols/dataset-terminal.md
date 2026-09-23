# Dataset: running and inspecting it

This page lists the {py:obj}`Dataset <batcher.Dataset>` methods that execute a plan or report on one: the terminal calls that hand back Arrow or write to a sink, the previews and counts, the schema and plan inspection, the column statistics, and the Python protocols a dataset answers to. The methods that only build a plan are on {doc}`dataset-transforms`.

A terminal call is the moment the optimizer runs, so the plan it sees is the whole chain. That is why `explain()` is listed here rather than with the transforms: it reports the optimized plan, which does not exist until something asks for it.

```{eval-rst}
.. currentmodule:: batcher
```

## Executing and exporting

These terminal operations execute the plan and hand the result to Arrow, Python, another framework, or a sink.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.collect
   Dataset.to_arrow
   Dataset.to_pydict
   Dataset.to_pylist
   Dataset.to_pandas
   Dataset.to_polars
   Dataset.to_numpy
   Dataset.to_jax
   Dataset.to_ray_dataset
   Dataset.to_spark
   Dataset.to_daft
   Dataset.write
```

## Previewing, counting and iterating

These terminal operations count, print, compare or stream the result instead of collecting all of it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.show
   Dataset.glimpse
   Dataset.first
   Dataset.last
   Dataset.item
   Dataset.count
   Dataset.is_empty
   Dataset.has_rows
   Dataset.equals
   Dataset.iter_batches
   Dataset.iter_rows
   Dataset.iter_slices
```

## Inspecting schema and plan

These members describe the output columns and the plan, and `explain` and `lineage` read the plan without executing it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.schema
   Dataset.columns
   Dataset.dtypes
   Dataset.collect_schema
   Dataset.width
   Dataset.shape
   Dataset.size
   Dataset.is_streaming
   Dataset.explain
   Dataset.lineage
```

## Column statistics

Each of these executes the query and returns one value computed over a column, such as a sum, a quantile or a correlation.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.count_distinct
   Dataset.approx_count_distinct
   Dataset.min
   Dataset.max
   Dataset.sum
   Dataset.mean
   Dataset.median
   Dataset.mode
   Dataset.product
   Dataset.std
   Dataset.var
   Dataset.quantile
   Dataset.approx_quantile
   Dataset.approx_median
   Dataset.approx_percentile
   Dataset.skew
   Dataset.kurtosis
   Dataset.mad
   Dataset.any
   Dataset.all
   Dataset.corr
   Dataset.cov
```

## Profiling and summaries

These methods summarize many columns at once, and `stats` returns the measured per-operator execution statistics.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.describe
   Dataset.info
   Dataset.profile
   Dataset.nunique
   Dataset.memory_usage
   Dataset.corr_matrix
   Dataset.cov_matrix
   Dataset.stats
```

## Caching and write layout

These methods cache a computed result, drop it again, or set how the next `write` lays out its files.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.cache
   Dataset.uncache
   Dataset.repartition
```

## Streaming

These methods declare event-time watermarks and the stateful operations that run over an unbounded source.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.with_watermark
   Dataset.drop_duplicates_within_watermark
   Dataset.join_stream
   Dataset.session_window
   Dataset.transform_with_state
```

## Accessors

Each property returns a namespace documented on {doc}`dataset-accessors`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.ml
   Dataset.dq
   Dataset.scd
   Dataset.meta
```

## Python protocols and operators

These special methods let a dataset be indexed, counted, iterated, tested for a column, combined with `+`, `|`, `&` and `-`, and exported over the Arrow PyCapsule stream interface.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.__getitem__
   Dataset.__len__
   Dataset.__iter__
   Dataset.__contains__
   Dataset.__add__
   Dataset.__or__
   Dataset.__and__
   Dataset.__sub__
   Dataset.__arrow_c_stream__
```

## See also

- {doc}`dataset-transforms`: the lazy methods that build the plan these run.
- {doc}`dataset-accessors`: the `ml`, `dq`, `scd`, and `meta` namespaces a dataset hands out.
- {doc}`/user-guide/operate/tuning/explain-plans`: how to read what `explain` prints.
- {doc}`/api/relational/dataset`: the same methods with the semantics behind each one.
