# Regular expressions over a column: extract, replace, and count

``extract`` pulls one capture group, ``extract_all`` returns a list column of every match, and ``replace_all`` rewrites every occurrence. The pattern is compiled once per operator rather than per row, which is the whole reason these live in the engine.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_regex.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_regex.py
```

## See also

- {doc}`strings_ratios`: character-class ratios: cheap quality signals for a text corpus.
- {doc}`strings_search`: string search: substring tests, multi-pattern tests, and match counting.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
