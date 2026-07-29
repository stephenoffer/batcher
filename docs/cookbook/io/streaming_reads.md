# Reading in bounded memory: iter_batches, limits, and lazy metadata

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
