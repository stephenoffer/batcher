# Dataset and its accessors

The {py:class}`Dataset <batcher.Dataset>` surface itself, the grouped form, and the
namespaces reached from a dataset for machine learning, data quality, slowly changing
dimensions, and metadata.

## Dataset

Every transformation below returns a new `Dataset` and runs nothing.

The terminal operations are the exception, and they are the ones that hand back Arrow or
write to a sink: {py:meth}`collect <batcher.Dataset.collect>`,
{py:meth}`to_pydict <batcher.Dataset.to_pydict>`,
{py:meth}`iter_batches <batcher.Dataset.iter_batches>`,
{py:obj}`write <batcher.Dataset.write>` and their siblings. Members are listed by group
rather than alphabetically, which at this size is the difference between a reference and a
word list.

```{eval-rst}
.. autoclass:: batcher.Dataset
   :members:
   :member-order: groupwise
   :special-members: __getitem__, __len__, __iter__, __contains__, __add__, __or__, __and__, __sub__, __arrow_c_stream__
```

## GroupBy

{py:meth}`ds.group_by(...) <batcher.Dataset.group_by>` returns this builder rather than a
dataset. It holds the keys and waits. `.agg(...)` names the output columns and hands a
`Dataset` back.

```{eval-rst}
.. autoclass:: batcher.GroupBy
   :members:
   :member-order: groupwise
```

## Dataset accessors

Three namespaces cover work the relational verbs have no spelling for.
{py:obj}`ds.ml <batcher.Dataset.ml>` runs inference and embeddings. It also holds the
loaders that hand a dataset to a training framework.
{py:obj}`ds.dq <batcher.Dataset.dq>` accumulates data-quality constraints until a terminal
call decides what to do with the rows that fail them. {py:obj}`ds.scd <batcher.Dataset.scd>`
treats the dataset as an incoming dimension snapshot and upserts it into a target table.

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

.. autoclass:: batcher.api.dataset.dq.ConstraintResult
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.scd.DatasetSCD
   :members:
   :member-order: bysource
```

## Metadata shortcuts

The {py:obj}`ds.meta <batcher.Dataset.meta>` namespace and the accessors it hands out read
their answers from Parquet footers, ORC stripe headers, lakehouse manifests and warehouse
catalogs rather than from the rows. A shortcut returns exactly what executing would return,
not an estimate of it. You rarely need the namespace, because the ordinary calls already
take the shortcut when one is available; reach for it to ask what the engine knows. The
{doc}`metadata shortcuts guide </user-guide/analyze/metadata-shortcuts>` says when it fires.

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

## See also

- {doc}`/api/relational/dataset`: the same methods taught with a runnable example per group.
- {doc}`/api/models/ml`: the `.ml` accessor at length, including the inference and embedding surfaces.
- {doc}`/user-guide/analyze/metadata-shortcuts`: when a call answers from metadata instead of scanning.
