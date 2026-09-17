# Fuzzy matching

Fuzzy matching finds the values that are almost a known string. Edit distances count operations, so lower is closer. The Jaro family returns a similarity between 0 and 1, so higher is closer. Pick by the error you expect: typos favor Levenshtein, transposed characters favor Damerau.

The comparison target is a plan-time literal, not another column. That makes these a fast screen against a known value such as a canonical name or a search term. The script compares `levenshtein`, `damerau_levenshtein`, `jaro_similarity`, `jaro_winkler_similarity`, and `hamming` against one name, keeps the near-matches above a threshold, and for two-table record linkage blocks on a `soundex` code and joins on that instead.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_similarity.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_similarity.py
```

## See also

- {doc}`/cookbook/expressions/strings/matching/strings_search`: substring tests, multi-pattern tests, and match counting.
- {doc}`/cookbook/expressions/strings/shaping/strings_slicing`: taking a fixed piece of every value.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
