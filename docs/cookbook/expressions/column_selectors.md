# Selectors: naming columns by type or pattern instead of one at a time

A selector is an ``Expr`` leaf standing for *every* matching column, so "round every float" is one expression that keeps working when a column is added. Spelling out names is how a pipeline silently stops covering a new column. Combine selectors with ``|``, ``&``, ``-``, and ``~``.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/column_selectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/column_selectors.py
```
