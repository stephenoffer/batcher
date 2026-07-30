# Reshaping: pivot, unpivot, explode, unnest, and set operations

Long-to-wide and back is the most common reshape in reporting. Pivot needs to know the value columns it will produce, which means it materializes; unpivot is the cheap direction and is usually what a downstream model actually wants.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/reshaping.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/reshaping.py
```

## See also

- {doc}`profiling`: profiling a table you have just been handed.
- {doc}`sampling_and_splits`: sampling and splitting: reproducible subsets that do not leak.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
