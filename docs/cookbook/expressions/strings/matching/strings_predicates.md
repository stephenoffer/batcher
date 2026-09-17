# String predicates

Running an LLM over a corpus costs money per row, so the cheapest win is not sending the rows that cannot help. These predicates all return a boolean column and compose with `&` and `|`, so a screen is one filter rather than a Python loop.

The script tests for blank, short, single-line, all-caps, and ASCII-only text, detects questions, digits, emails, phone numbers, and JSON-looking values, and guards an LLM budget with `estimate_tokens` and `fits_token_budget`. It combines them into one screen that keeps real questions and drops anything blank or shouted.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_predicates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_predicates.py
```

## See also

- {doc}`/cookbook/expressions/strings/measuring/strings_paths`: parsing file paths held in a column.
- {doc}`/cookbook/expressions/strings/measuring/strings_ratios`: cheap quality signals for a text corpus.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
