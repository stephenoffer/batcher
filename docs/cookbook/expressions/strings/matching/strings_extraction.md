# String extraction

Free text carries entities you want as columns: the emails, URLs, and numbers in a message, or its first sentence as a preview. The `extract_*` family returns a *list column*, so one row can carry many matches and you can `explode` it into one row per match. The `first_*`, `last_*`, and `truncate_*` family returns a scalar string, which is what you want for a preview or a title.

The script extracts emails, URLs, and numbers, builds previews with `first_sentence`, `first_word`, `truncate_words`, and `left`, and explodes the emails into a lookup table. One edge to plan for: a match that ends a sentence keeps its trailing period, so the script strips it with `trim(".")`.

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
