# Time zones

Store UTC, convert at the edge. A daily rollup computed in UTC and labeled as local time is wrong by up to a day at the boundary, and it is wrong quietly: the numbers look plausible, they are just attributed to the wrong day.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_timezones.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_timezones.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_parts`: pulling calendar parts out of a timestamp column.
- {doc}`/cookbook/expressions/temporal/temporal_truncation`: truncate to a period, or snap to a period boundary.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
