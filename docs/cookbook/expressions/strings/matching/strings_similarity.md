# Fuzzy matching

Edit distances count operations (lower is closer); the Jaro family returns a similarity in [0, 1] (higher is closer). Pick by the error you expect: typos favour Levenshtein, transposed characters favour Damerau.

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
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
