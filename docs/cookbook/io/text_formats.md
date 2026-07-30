# CSV, JSON, and Arrow IPC round trips

Text formats carry no schema, so types are inferred on read. That inference is the usual source of a surprise: a zip code column of "01234" becomes an integer and loses the leading zero. Read it, check the schema, and cast at the edge.

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

- {doc}`streaming_reads`: reading in bounded memory: iter_batches, limits, and lazy metadata.
- {doc}`sources_and_sinks`: the source and sink registries: what formats exist, and the objects behind them.
- {doc}`../../user-guide/reading-data`: every source format and how paths and schemas resolve.
- {doc}`../../user-guide/writing-data`: sinks, save modes, and partitioned output.
