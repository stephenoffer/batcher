# Time differences

``*_between`` gives a whole-unit difference between two columns, which is how you compute an age, a lead time, or a session length. ``offset_by`` shifts by a duration string, which is how you build a "30 days ago" cutoff without leaving the expression API.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_differences.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_differences.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_business_days`: weekend and business-day predicates, and formatting a timestamp for output.
- {doc}`/cookbook/expressions/temporal/temporal_parts`: pulling calendar parts out of a timestamp column.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
