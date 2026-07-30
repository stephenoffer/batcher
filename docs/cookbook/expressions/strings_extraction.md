# Pulling entities and leading fragments out of free text

The ``extract_*`` family returns a *list column*, so one row can carry many matches and you can ``explode`` it into one row per match. The ``first_*``/``last_*``/``truncate_*`` family returns a scalar string, which is what you want for a preview or a title.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_extraction.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_extraction.py
```

## See also

- {doc}`strings_counts`: counting structure in text: words, lines, sentences, and entities.
- {doc}`strings_hashing`: hashing and encoding a string column: keys, checksums, and safe transport.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
