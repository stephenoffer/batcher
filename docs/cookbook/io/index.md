# IO cookbook

This section holds 6 runnable recipes for getting data in and out, ordered from the format you will reach for first to the registry behind all of them.

Every page embeds a complete, self-contained script from the [`examples/io/`](https://github.com/batcher/batcher/tree/main/examples/io) directory. The scripts build their own in-memory data and assert on their own output, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`parquet_roundtrip` | Writing and reading Parquet, with partitioning and column pruning |
| {doc}`text_formats` | CSV, JSON, and Arrow IPC round trips |
| {doc}`save_modes` | What happens when the target already exists, and what a manifest records |
| {doc}`streaming_reads` | {py:meth}`iter_batches <batcher.Dataset.iter_batches>`, limits, and lazy metadata, in bounded memory |
| {doc}`arrow_interop` | Moving data in and out of other frameworks, zero-copy where possible |
| {doc}`sources_and_sinks` | Which formats exist, and the objects behind them |

## See also

- {doc}`/user-guide/moving-data/reading-data` and {doc}`/user-guide/moving-data/writing-data`: the guides these recipes condense.
- {doc}`/api/relational/io`: every reader and writer, with the optional extras each needs.
- {doc}`/user-guide/moving-data/custom-connectors`: registering a format of your own.
- {doc}`../../integrations/index`: connecting a specific system rather than a file format.

```{toctree}
:hidden:

parquet_roundtrip
text_formats
save_modes
streaming_reads
arrow_interop
sources_and_sinks
```
