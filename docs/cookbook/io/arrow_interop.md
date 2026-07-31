# Arrow interop

Arrow is the shared contract, so ``from_arrow``/``to_arrow`` are the cheapest boundary there is. The pandas and Polars bridges go through Arrow too, which is why they are much cheaper than a row-by-row conversion.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/arrow_interop.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/arrow_interop.py
```

## See also

- {doc}`parquet_roundtrip`: writing and reading Parquet, with partitioning and column pruning.
- {doc}`save_modes`: what happens when the target already exists.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
