# Building a Dataset

This page lists the top-level functions that build a {py:obj}`Dataset <batcher.Dataset>` from data already in the process: a dict, a list of rows, an Arrow table, another engine's frame, or a generated range. Reading from a path or an external system is on {doc}`readers-and-writers`.

Every one of these is zero-copy where the source format allows it. `from_arrow` and `from_pandas` hand the buffers straight to the engine rather than converting row by row, which is the difference between a handle and an import.

The `from_*` family wraps an object already in the process, and `read` is the entry point
for files, tables, and streams.

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
   from_pylist
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
   from_daft
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   from_any
   read
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

## See also

- {doc}`readers-and-writers`: reading from a path, a table, a database, or a stream.
- {doc}`/api/relational/io`: the same surface with a runnable example per source family.
- {doc}`/getting-started/migration/index`: the equivalent call in the engine you are coming from.
