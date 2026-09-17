# Save modes and manifests

`ds.write` defaults to `mode="overwrite"`, so writing to a path that already exists replaces it. Pass `mode="error"` when a retried job must not clobber earlier output, or `mode="ignore"` to skip the write.

`mode="append"` raises a `PlanError` on a plain file sink, because there is no table to add to. The script shows the two alternatives: one file per batch under a directory, read back as one relation, or a transactional Delta, Iceberg, or Hudi sink with a real append. Every write returns a `WriteManifest` describing what it produced, which is what you record for lineage or a resume.

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
