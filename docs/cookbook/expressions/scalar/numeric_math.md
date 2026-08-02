# Numeric math

All of these are columnar and fuse into a single pass, so a chain of ten of them is not ten scans. Watch the division operators in particular: ``/`` is true division and ``floordiv`` truncates, and mixing them up is a quiet source of off-by-one bugs.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/numeric_math.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/numeric_math.py
```

## See also

- {doc}`/cookbook/expressions/scalar/nulls_and_casting`: the two places a pipeline quietly changes its answer.
- {doc}`/cookbook/expressions/scalar/sorting_and_ranking`: sorting and ranking, including the edge cases that hide bugs.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
