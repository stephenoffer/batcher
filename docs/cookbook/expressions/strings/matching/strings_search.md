# String search

Every predicate here is a columnar expression, so the whole column is tested in Rust rather than one Python call per row. `contains_any` and `contains_all` take a list of patterns and fold to a single boolean column, which is what you want for a keyword screen: one pass, not one pass per keyword.

The script runs plain substring, prefix, and suffix tests, the multi-keyword tests, and counts occurrences with `count_matches` for a pattern and `count_char` for a single character.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_search.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_search.py
```

## See also

- {doc}`/cookbook/expressions/strings/matching/strings_regex`: extract, replace, and count.
- {doc}`/cookbook/expressions/strings/matching/strings_similarity`: fuzzy string matching against a reference value.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
