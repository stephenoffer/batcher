# Dataset accessors

This page is the complete reference for the namespaces a {py:class}`Dataset <batcher.Dataset>` carries for machine learning, data quality, slowly changing dimensions and metadata. The dataset itself is on {doc}`dataset`.

Three namespaces cover work the relational verbs have no spelling for.
{py:obj}`ds.ml <batcher.Dataset.ml>` runs inference and embeddings. It also holds the
loaders that hand a dataset to a training framework.
{py:obj}`ds.dq <batcher.Dataset.dq>` accumulates data-quality constraints until a terminal
call decides what to do with the rows that fail them. {py:obj}`ds.scd <batcher.Dataset.scd>`
treats the dataset as an incoming dimension snapshot and upserts it into a target table.

## Machine learning: ds.ml

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autoclass:: DatasetML
   :no-members:
```

### Inference and generation

These methods run a model over the rows: batch inference, embeddings, tabular scoring, and LLM generation, classification and extraction.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetML.infer
   DatasetML.embed
   DatasetML.predict
   DatasetML.generate
   DatasetML.classify
   DatasetML.extract
```


### Embeddings and vector search

These methods prepare an embedding column and retrieve, join or fuse rows by embedding similarity.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetML.normalize_embeddings
   DatasetML.truncate_embeddings
   DatasetML.binarize_embeddings
   DatasetML.drop_degenerate_embeddings
   DatasetML.similarity_to
   DatasetML.nearest_neighbors
   DatasetML.batched_nearest_neighbors
   DatasetML.similarity_join
   DatasetML.reciprocal_rank_fusion
```


### Downloading, uploading and deduplicating

These methods fetch or write the bytes behind each row, and find or remove near-duplicate documents.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetML.download
   DatasetML.upload
   DatasetML.near_duplicates
   DatasetML.drop_near_duplicates
```


### Splitting and evaluating

These methods split rows for training and validation, score predictions and retrieval results, and compare feature distributions.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetML.train_test_split
   DatasetML.random_split
   DatasetML.kfold
   DatasetML.time_series_split
   DatasetML.evaluate
   DatasetML.recall_at_k
   DatasetML.mrr
   DatasetML.drift
```


### Training loaders

These methods stream a dataset to a training framework, or write it as a shard corpus to stream later.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetML.to_torch_dataloader
   DatasetML.iter_torch_batches
   DatasetML.stream_loader
   DatasetML.to_tf
   DatasetML.to_numpy_batches
   DatasetML.write_shards
```


## Data quality: ds.dq

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autoclass:: DatasetDQ
   :no-members:
```

### Acting on the results

These terminal calls execute the accumulated constraints and decide what happens to the rows that fail them.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetDQ.validate
   DatasetDQ.fail
   DatasetDQ.drop
   DatasetDQ.quarantine
   DatasetDQ.annotate
```


### Row value constraints

Each of these requires every row's value in a column to satisfy a condition.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetDQ.not_null
   DatasetDQ.in_range
   DatasetDQ.accepted_values
   DatasetDQ.rejected_values
   DatasetDQ.matches
   DatasetDQ.not_matches
   DatasetDQ.matches_format
   DatasetDQ.not_empty
   DatasetDQ.positive
   DatasetDQ.is_finite
   DatasetDQ.str_length_between
   DatasetDQ.not_in_future
   DatasetDQ.compare_columns
```


### Aggregate constraints

Each of these requires a statistic computed over the whole relation or a column to fall inside a bound.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetDQ.row_count_between
   DatasetDQ.mean_between
   DatasetDQ.median_between
   DatasetDQ.stddev_between
   DatasetDQ.sum_between
   DatasetDQ.quantile_between
   DatasetDQ.distinct_count_between
   DatasetDQ.null_rate_below
   DatasetDQ.unique_ratio_above
   DatasetDQ.fresh_within
```


### Schema, keys and references

These constraints check the column set and types, uniqueness, and whether key values resolve in another dataset.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetDQ.has_columns
   DatasetDQ.column_types
   DatasetDQ.no_unexpected_columns
   DatasetDQ.unique
   DatasetDQ.references
   DatasetDQ.foreign_key
```


### Scoping and custom constraints

These calls add a custom predicate, scope later constraints to a subset of rows, rebind the chain, or propose constraints the data already satisfies.

```{eval-rst}
.. currentmodule:: batcher.api.dataset.dq

.. autosummary::
   :toctree: generated
   :nosignatures:

   DatasetDQ.check
   DatasetDQ.where
   DatasetDQ.on
   DatasetDQ.suggest
```


### Validation results

{py:meth}`validate <batcher.api.dataset.dq.DatasetDQ.validate>` returns a report holding one result per constraint.

```{eval-rst}
.. currentmodule:: batcher.api.dataset

.. autosummary::
   :toctree: generated
   :nosignatures:

   dq.ValidationReport
   dq.ConstraintResult
```

## Slowly changing dimensions: ds.scd

The accessor class has its own page, which lists each of its methods.

```{eval-rst}
.. currentmodule:: batcher.api.dataset

.. autosummary::
   :toctree: generated
   :nosignatures:

   scd.DatasetSCD
```

## Metadata shortcuts

The {py:obj}`ds.meta <batcher.Dataset.meta>` namespace and the accessors it hands out read
their answers from Parquet footers, ORC stripe headers, lakehouse manifests and warehouse
catalogs rather than from the rows. A shortcut returns exactly what executing would return,
not an estimate of it. You rarely need the namespace, because the ordinary calls already
take the shortcut when one is available. Reach for it to ask what the engine knows. The
{doc}`metadata shortcuts guide </user-guide/analyze/metadata-shortcuts>` says when it fires.

```{eval-rst}
.. currentmodule:: batcher.api.dataset

.. autosummary::
   :toctree: generated
   :nosignatures:

   meta.frame.DatasetMeta
   meta.column.ColumnMeta
   meta.checks.ColumnChecks
   meta.schema.SchemaMeta
   meta.nulls.NullsMeta
   meta.approx.ApproxMeta
   meta.storage.StorageMeta
   meta.pair.PairMeta
```

## See also

- {doc}`dataset`: the `Dataset` and `GroupBy` members these namespaces hang off.
- {doc}`/api/models/ml`: the `.ml` accessor at length, including the inference and embedding surfaces.
- {doc}`/user-guide/analyze/metadata-shortcuts`: when a call answers from metadata instead of scanning.
