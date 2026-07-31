# Save modes and manifests

The default refuses to clobber, which is the safe choice for a job that might be retried. ``overwrite`` replaces, ``append`` adds. Every write returns a manifest describing what it actually produced, which is what you record for lineage or resume.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/save_modes.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/save_modes.py
```

## See also

- {doc}`parquet_roundtrip`: writing and reading Parquet, with partitioning and column pruning.
- {doc}`sources_and_sinks`: what formats exist, and the objects behind them.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
