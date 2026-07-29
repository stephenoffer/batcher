# Bounded memory: caching a reused branch and spilling under a tight budget

``cache()`` is an execution hint, not a semantic change: the result is identical with or without it. Spilling is the same idea for memory -- under a small budget the engine goes out of core rather than failing, and the answer does not change.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/memory_and_caching.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/memory_and_caching.py
```
