# Fuzzy string matching against a reference value

Edit distances count operations (lower is closer); the Jaro family returns a similarity in [0, 1] (higher is closer). Pick by the error you expect: typos favour Levenshtein, transposed characters favour Damerau.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_similarity.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_similarity.py
```

## See also

- {doc}`strings_search`: string search: substring tests, multi-pattern tests, and match counting.
- {doc}`strings_slicing`: string slicing: taking a fixed piece of every value.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
