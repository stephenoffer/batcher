# Pulling calendar parts out of a timestamp column

Every accessor here is a projection, so extracting a year to group by costs one pass and no Python. The SQL spellings (``dayofweek``, ``weekofyear``, ``monthname``) and the Polars spellings (``weekday``, ``week``, ``month_name``) both exist and agree.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/temporal_parts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_parts.py
```

## See also

- {doc}`temporal_differences`: durations between two timestamp columns, and shifting a timestamp.
- {doc}`temporal_timezones`: time zones: converting between them, and the reporting-boundary trap.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
