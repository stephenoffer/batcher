# Parquet round trip

Parquet is the format to reach for first. Its footer carries row counts and column statistics, so `count()` is answered without decoding any data, and a read that selects one column never touches the others.

The script writes a table, reads it back, then writes it again with `partition_by=["day"]`. A filter on `day` now prunes whole directories. Expect one change on the way back: the partition value is parsed out of the directory name, so the `day` strings read back as dates.

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
- {doc}`save_modes`: what happens when the target already exists.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
