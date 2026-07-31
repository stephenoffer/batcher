# Source and sink registries

``bt.read.parquet(...)`` is a façade over a registry of ``SourceFormat`` implementations. Reading the registry is how you discover what is supported in *this* build, rather than trusting a docs page that may predate an extra you have not installed.

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
