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
