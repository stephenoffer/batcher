# Writing and reading Parquet, with partitioning and column pruning

Parquet is the default for a reason: the footer carries statistics, so a filtered read skips row groups without decoding them and ``count()`` is answered from metadata alone. Partitioning on a column you always filter by turns that skipping into directory pruning.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/parquet_roundtrip.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/parquet_roundtrip.py
```

## See also

- {doc}`arrow_interop`: moving data in and out of other frameworks, zero-copy where possible.
- {doc}`save_modes`: save modes and write manifests: what happens when the target already exists.
- {doc}`../../user-guide/reading-data`: every source format and how paths and schemas resolve.
- {doc}`../../user-guide/writing-data`: sinks, save modes, and partitioned output.
