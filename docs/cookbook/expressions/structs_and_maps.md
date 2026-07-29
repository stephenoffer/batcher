# Struct and map columns: nested records without flattening the table

A struct column holds a fixed set of named fields per row; a map column holds variable key/value pairs. Both are read with an accessor rather than by exploding the table, so a nested field stays one projection away.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/structs_and_maps.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/structs_and_maps.py
```
