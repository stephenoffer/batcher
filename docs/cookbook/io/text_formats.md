# Text formats

CSV and JSON carry no schema, so Batcher infers the types on read. That inference is the usual source of surprise, and the script makes it concrete: a zip code column holding `"01234"` round-trips through CSV as the integer `1234`.

Arrow IPC records the type beside the data, so the same column comes back as the string it was. When a text format is unavoidable, check the schema right after the read, before a lost leading zero reaches anything downstream.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/text_formats.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/text_formats.py
```

## See also

- {doc}`streaming_reads`: iter_batches, limits, and lazy metadata.
- {doc}`sources_and_sinks`: what formats exist, and the objects behind them.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
