# Sorting and ranking, including the edge cases that hide bugs

Sort order is where nulls, ties, and descending flags interact badly. Decide explicitly where nulls go and how ties break, because the default is rarely what a report wants and the difference is invisible until someone checks a boundary row.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/sorting_and_ranking.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/sorting_and_ranking.py
```

## See also

- {doc}`numeric_math`: arithmetic and math functions on numeric columns.
- {doc}`strings_case`: string case: normalizing capitalization before you compare or group.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
