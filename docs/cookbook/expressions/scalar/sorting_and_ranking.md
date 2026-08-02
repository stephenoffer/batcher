# Sorting and ranking

Sort order is where nulls, ties, and descending flags interact badly. Decide explicitly where nulls go and how ties break, because the default is rarely what a report wants and the difference is invisible until someone checks a boundary row.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/sorting_and_ranking.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/sorting_and_ranking.py
```

## See also

- {doc}`/cookbook/expressions/scalar/numeric_math`: arithmetic and math functions on numeric columns.
- {doc}`/cookbook/expressions/strings/shaping/strings_case`: normalizing capitalization before you compare or group.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
