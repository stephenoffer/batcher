# IO cookbook

Reading and writing: Parquet, text formats, Arrow interop, save modes, streaming reads, and the source/sink registries.

Every page here embeds a complete, self-contained script from the
[`examples/io/`](https://github.com/batcher/batcher/tree/main/examples/io) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`arrow_interop` | Moving data in and out of other frameworks, zero-copy where possible |
| {doc}`parquet_roundtrip` | Writing and reading Parquet, with partitioning and column pruning |
| {doc}`save_modes` | Save modes and write manifests: what happens when the target already exists |
| {doc}`sources_and_sinks` | The source and sink registries: what formats exist, and the objects behind them |
| {doc}`streaming_reads` | Reading in bounded memory: iter_batches, limits, and lazy metadata |
| {doc}`text_formats` | CSV, JSON, and Arrow IPC round trips |

```{toctree}
:hidden:

arrow_interop
parquet_roundtrip
save_modes
sources_and_sinks
streaming_reads
text_formats
```
