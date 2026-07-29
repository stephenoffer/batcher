# Parsing file paths held in a column

Object-storage listings arrive as one long URI per row. Splitting them in the engine keeps the partition key, the extension, and the directory available as ordinary columns you can group and filter by.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_paths.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_paths.py
```
