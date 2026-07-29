# The source and sink registries: what formats exist, and the objects behind them

``bt.read.parquet(...)`` is a façade over a registry of ``SourceFormat`` implementations. Reading the registry is how you discover what is supported in *this* build, rather than trusting a docs page that may predate an extra you have not installed.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/sources_and_sinks.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/sources_and_sinks.py
```
