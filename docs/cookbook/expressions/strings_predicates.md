# Boolean text predicates: the screen in front of an expensive stage

Running an LLM over a corpus costs money per row, so the cheapest win is not sending the rows that cannot help. These predicates all return a boolean column and compose with ``&``/``|``, so a screen is one filter rather than a Python loop.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_predicates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_predicates.py
```

## See also

- {doc}`strings_paths`: parsing file paths held in a column.
- {doc}`strings_ratios`: character-class ratios: cheap quality signals for a text corpus.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
