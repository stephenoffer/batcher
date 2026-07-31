# Calendar parts

Every accessor here is a projection, so extracting a year to group by costs one pass and no Python. The SQL spellings (``dayofweek``, ``weekofyear``, ``monthname``) and the Polars spellings (``weekday``, ``week``, ``month_name``) both exist and agree.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_parts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_parts.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_differences`: durations between two timestamp columns, and shifting a timestamp.
- {doc}`/cookbook/expressions/temporal/temporal_timezones`: converting between them, and the reporting-boundary trap.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
