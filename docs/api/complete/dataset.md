# Dataset and GroupBy

This page is the complete reference for the {py:class}`Dataset <batcher.Dataset>` surface and the {py:class}`GroupBy <batcher.GroupBy>` builder. The namespaces reached from a dataset for machine learning, data quality, slowly changing dimensions and metadata are on {doc}`dataset-accessors`.

## Dataset

Every transformation below returns a new `Dataset` and runs nothing.

The terminal operations are the exception, and they are the ones that hand back Arrow or
write to a sink: {py:meth}`collect <batcher.Dataset.collect>`,
{py:meth}`to_pydict <batcher.Dataset.to_pydict>`,
{py:meth}`iter_batches <batcher.Dataset.iter_batches>`,
{py:obj}`write <batcher.Dataset.write>` and their siblings. Members are listed by group
rather than alphabetically, because an alphabetical list this long is a word list. Each
member has its own page.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: Dataset
   :no-members:
```

### Selecting and deriving columns

These methods choose, add, rename, retype and transform columns while the rows stay as they are.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.select
   Dataset.with_columns
   Dataset.drop
   Dataset.rename
   Dataset.cast
   Dataset.select_dtypes
   Dataset.with_row_index
   Dataset.with_random
   Dataset.unnest
   Dataset.drop_constant_columns
   Dataset.round
   Dataset.abs
   Dataset.clip
```


### Filtering and sampling rows

These methods keep a subset of the rows, by predicate or by a reproducible random draw.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.filter
   Dataset.sample
   Dataset.shuffle
   Dataset.gather_every
   Dataset.sample_per_group
```


### Sorting and limiting

These methods order the rows or keep a bounded number of them.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.sort
   Dataset.limit
   Dataset.tail
   Dataset.top_k
   Dataset.bottom_k
   Dataset.reverse
```


### Joins and set operations

These methods combine a dataset with another one, by key, by nearest match, by lookup, or as a set.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.join
   Dataset.cross_join
   Dataset.join_asof
   Dataset.lookup_join
   Dataset.union
   Dataset.intersect
   Dataset.except_
```


### Grouping and aggregation

{py:meth}`group_by <batcher.Dataset.group_by>` returns the {py:class}`GroupBy <batcher.GroupBy>` builder below, and the rest aggregate the whole dataset or several grouping levels at once.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.group_by
   Dataset.agg
   Dataset.rollup
   Dataset.cube
   Dataset.grouping_sets
   Dataset.value_counts
   Dataset.crosstab
```


### Windows and reshaping

These methods append window-function columns or change the shape of the table between long and wide.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.window
   Dataset.pivot
   Dataset.unpivot
   Dataset.explode
   Dataset.get_dummies
```


### Nulls and duplicates

These methods drop or fill nulls, remove duplicate rows, and report where nulls are.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.drop_nulls
   Dataset.fill_null
   Dataset.distinct
   Dataset.isna
   Dataset.notna
   Dataset.null_count
   Dataset.n_null
   Dataset.has_nulls
   Dataset.all_null
```


### Executing and exporting

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


### Previewing, counting and iterating

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


### Inspecting schema and plan

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


### Column statistics

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


### Profiling and summaries

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


### SQL, UDFs and batch maps

These methods run a Python function over batches or rows, or run a SQL query with this dataset bound as a table.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.map_batches
   Dataset.map
   Dataset.flat_map
   Dataset.pipe
   Dataset.sql
```


### Caching and write layout

These methods cache a computed result, drop it again, or set how the next `write` lays out its files.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.cache
   Dataset.uncache
   Dataset.repartition
```


### Streaming

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


### ML pipeline helpers

These methods split, balance and filter a dataset for training, and move large payloads out of the rows and back.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.stratified_split
   Dataset.train_val_test_split
   Dataset.split_at_indices
   Dataset.split_proportionately
   Dataset.balance_classes
   Dataset.class_balance
   Dataset.class_weights
   Dataset.filter_by_length
   Dataset.filter_by_token_budget
   Dataset.drop_empty
   Dataset.offload_blobs
   Dataset.materialize_blobs
```


### Accessors

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


### Python protocols and operators

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


## GroupBy

{py:meth}`ds.group_by(...) <batcher.Dataset.group_by>` returns this builder rather than a
dataset. It holds the keys and waits. `.agg(...)` names the output columns and hands a
`Dataset` back.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: GroupBy
   :no-members:
```

### Common aggregates

`agg` takes any aggregate expressions, and the rest are shorthands that apply one aggregate to every value column per group.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.agg
   GroupBy.len
   GroupBy.count
   GroupBy.count_distinct
   GroupBy.sum
   GroupBy.mean
   GroupBy.min
   GroupBy.max
   GroupBy.first
   GroupBy.last
```


### Statistical and collecting aggregates

These shorthands compute a distribution statistic per group, or collect each group's values into a list.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.median
   GroupBy.mode
   GroupBy.quantile
   GroupBy.std
   GroupBy.var
   GroupBy.skew
   GroupBy.kurtosis
   GroupBy.product
   GroupBy.array_agg
```


### Rows and functions per group

These members keep the first or last rows of each group, apply a Python function to each group, or report the keys.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   GroupBy.head
   GroupBy.tail
   GroupBy.map_groups
   GroupBy.keys
   GroupBy.__iter__
```


## See also

- {doc}`dataset-accessors`: the `ml`, `dq`, `scd` and `meta` namespaces a dataset carries.
- {doc}`/api/relational/dataset`: the same methods taught with a runnable example per group.
- {doc}`/api/models/ml`: the `.ml` accessor at length, including the inference and embedding surfaces.
- {doc}`/user-guide/analyze/metadata-shortcuts`: when a call answers from metadata instead of scanning.
