# Numeric math

All of these are columnar and fuse into a single pass, so a chain of ten of them is not ten scans. Watch the division operators. `/` is true division, and `//` rounds toward negative infinity rather than toward zero, so `-7 // 2` is `-4` where SQL integer division gives `-3`. Mixing the two up is a quiet source of off-by-one bugs on negative values.

The script covers `abs`, `sign`, rounding, `floor` and `ceil`, `sqrt`, `exp`, and `**`, then `clip` for bounding a column without a branch, division and modulo, the `between` and `is_in` range and membership predicates, and bitwise operations on integers.

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
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
