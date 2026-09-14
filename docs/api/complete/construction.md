# Construction and I/O

Every way to build a {py:class}`Dataset <batcher.Dataset>`, every way to get data back out,
and the two namespaces those calls hand you.

## Top-level functions

The module-level names `batcher` exports, less the expressions, the configuration
functions and governance, which have pages of their own. Most of them construct a dataset.
The `from_*` family wraps an object already in the process; the `read_*` shorthands name a
path or a registered table.

The rest is a mixed bag. Nothing in it hangs off a `Dataset`, so the list is worth reading
rather than guessing at: {py:func}`compact <batcher.compact>` and
{py:func}`vacuum <batcher.vacuum>` maintain a transactional table and take a path rather
than a dataset, {py:func}`streams <batcher.streams>`,
{py:func}`await_any_termination <batcher.await_any_termination>` and
{py:func}`reset_terminated <batcher.reset_terminated>` control running streaming queries
under Spark's names, and {py:func}`register_function <batcher.register_function>` and
{py:func}`register_model <batcher.register_model>` extend the default SQL session that
{py:func}`bt.sql <batcher.sql>` falls back on. {py:func}`versions <batcher.versions>`
reports the build, compiled engine included.

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
   reset_terminated
   register_function
   register_model
   udf
   compact
   vacuum
   release_cluster
   engine_version
   versions
   show_versions
   accelerators
   show_accelerators
   measure_energy
   start_ui
   stop_ui
   ui_url
```

## Reading and writing

The format-specific calls hang off two namespaces rather than the top level.
{py:obj}`bt.read <batcher.read>` holds every reader and returns a lazy dataset: nothing is
opened yet. {py:obj}`ds.write <batcher.Dataset.write>` holds every writer and is terminal,
so it runs the plan and hands back a manifest of the files it wrote.

```{eval-rst}
.. autoclass:: batcher.api.io_namespace.reader.Reader
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.io_namespace.writer.Writer
   :members:
   :member-order: bysource
```

## See also

- {doc}`/api/relational/io`: the same readers and writers with their options, save modes, and the extras each connector installs.
- {doc}`dataset`: what a `Dataset` does once you hold one.
- {doc}`/user-guide/moving-data/reading-data`: choosing a reader, and the cloud paths and credentials behind it.
