# String search

Every predicate here is a columnar expression, so the whole column is tested in Rust rather than one Python call per row. ``contains_any``/``contains_all`` take an iterable of patterns and fold to a single boolean column, which is what you want for a keyword screen: one pass, not one pass per keyword.

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
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
