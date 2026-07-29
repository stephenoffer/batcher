# Deduplication: exact keys, whole rows, and keeping a chosen survivor

"Remove duplicates" is under-specified until you say *which* copy survives. Keeping an arbitrary one is how a pipeline becomes non-deterministic; keeping the latest by a timestamp is almost always what was meant.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/deduplication.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/deduplication.py
```
