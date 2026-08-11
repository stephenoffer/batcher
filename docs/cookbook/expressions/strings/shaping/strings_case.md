# String case

Case folding is the cheapest way to stop a group_by splitting "ACME", "Acme", and "acme" into three groups. Do it once in the projection, then group on the normalized column.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_case.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_case.py
```

## See also

- {doc}`/cookbook/expressions/scalar/sorting_and_ranking`: sorting and ranking, including the edge cases that hide bugs.
- {doc}`/cookbook/expressions/strings/shaping/strings_chunking`: splitting long documents into overlapping chunks for a RAG index.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
