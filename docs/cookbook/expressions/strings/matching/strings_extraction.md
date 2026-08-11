# String extraction

The ``extract_*`` family returns a *list column*, so one row can carry many matches and you can ``explode`` it into one row per match. The ``first_*``/``last_*``/``truncate_*`` family returns a scalar string, which is what you want for a preview or a title.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_extraction.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_extraction.py
```

## See also

- {doc}`/cookbook/expressions/strings/measuring/strings_counts`: words, lines, sentences, and entities.
- {doc}`/cookbook/expressions/strings/measuring/strings_hashing`: keys, checksums, and safe transport.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
