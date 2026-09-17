# Reading in bounded memory

`collect()` materializes the whole result. `iter_batches()` streams Arrow `RecordBatch`es through the pipeline instead, so memory is bounded by the batch size rather than the table, and a table larger than memory still works.

The script streams a Parquet file 128 rows at a time and shows that a filter and a `limit` both cut what the stream reads. Some questions need no data at all: `count()` and the schema come from file metadata.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/streaming_reads.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/streaming_reads.py
```

## See also

- {doc}`sources_and_sinks`: what formats exist, and the objects behind them.
- {doc}`text_formats`: CSV, JSON, and Arrow IPC round trips.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
