# Dataset: building a plan

This page lists the {py:obj}`Dataset <batcher.Dataset>` methods that return a new dataset and run nothing. Each one appends a node to a logical plan, so a chain of forty of them costs forty object allocations and no data movement. The methods that execute that plan are on {doc}`dataset-terminal`.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: Dataset
   :no-members:
```

## Selecting and deriving columns

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
   Dataset.match_to_schema
   Dataset.drop_constant_columns
   Dataset.round
   Dataset.abs
   Dataset.clip
```

## Filtering and sampling rows

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

## Sorting and limiting

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

## Joins and set operations

These methods combine a dataset with another one: by key, by an arbitrary predicate, by nearest match, by lookup, by position, or as a set. {py:obj}`update <batcher.Dataset.update>` is the odd one out, overwriting matched rows in place of adding columns.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.join
   Dataset.cross_join
   Dataset.join_asof
   Dataset.join_where
   Dataset.lookup_join
   Dataset.update
   Dataset.zip
   Dataset.union
   Dataset.intersect
   Dataset.except_
```

## Grouping and aggregation

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

## Windows, reshaping, and splitting

These methods append window-function columns, change the shape of the table between long and wide, or divide one dataset into several. {py:obj}`partition_by <batcher.Dataset.partition_by>` and {py:obj}`split <batcher.Dataset.split>` each hand back a collection of datasets, and like everything else here they run nothing until one of them is collected.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.window
   Dataset.pivot
   Dataset.unpivot
   Dataset.transpose
   Dataset.explode
   Dataset.get_dummies
   Dataset.partition_by
   Dataset.split
```

## Nulls and duplicates

These methods drop or fill nulls, drop NaNs, remove duplicate rows, and report where the nulls are. Null and NaN are separate here, as they are in Arrow: a float column can hold both, and {py:obj}`drop_nulls <batcher.Dataset.drop_nulls>` leaves a NaN alone.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Dataset.drop_nulls
   Dataset.drop_nans
   Dataset.fill_null
   Dataset.distinct
   Dataset.isna
   Dataset.notna
   Dataset.null_count
   Dataset.n_null
   Dataset.has_nulls
   Dataset.all_null
```

## SQL, UDFs and batch maps

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

## ML pipeline helpers

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

## See also

- {doc}`dataset-terminal`: the calls that execute the plan these build.
- {doc}`groupby`: the {py:obj}`GroupBy <batcher.GroupBy>` that `group_by` returns.
- {doc}`/api/relational/dataset`: the same methods with the semantics behind each one.
- {doc}`/user-guide/transform/index`: the guides these methods are the reference for.
