# Source and sink registries

`bt.read.parquet(...)` and `ds.write.parquet(...)` are thin wrappers over two registries, `SOURCES` and `SINKS` in `batcher.io`. Listing them tells you what *this* build can read and write, including the formats an optional extra adds.

The script lists both registries and looks up the `ParquetSource` and `ParquetSink` classes behind the readers. Those classes are the contract to study when you write a connector of your own.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/sources_and_sinks.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/sources_and_sinks.py
```

## See also

- {doc}`save_modes`: what happens when the target already exists.
- {doc}`streaming_reads`: iter_batches, limits, and lazy metadata.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
