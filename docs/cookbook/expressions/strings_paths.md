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

## See also

- {doc}`strings_padding`: string padding and trimming: fixed-width keys and cleaning stray whitespace.
- {doc}`strings_predicates`: boolean text predicates: the screen in front of an expensive stage.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
