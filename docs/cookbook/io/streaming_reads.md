# Reading in bounded memory

``collect()`` materializes. ``iter_batches()`` does not: it streams Arrow batches through the pipeline so a table larger than memory still works. The metadata shortcuts go further and answer some questions without reading any data at all.

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
