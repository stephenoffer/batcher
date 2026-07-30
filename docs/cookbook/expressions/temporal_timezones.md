# Time zones: converting between them, and the reporting-boundary trap

Store UTC, convert at the edge. A daily rollup computed in UTC and labelled as local time is wrong by up to a day at the boundary, and it is wrong quietly -- the numbers look plausible, they are just attributed to the wrong day.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/temporal_timezones.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_timezones.py
```

## See also

- {doc}`temporal_parts`: pulling calendar parts out of a timestamp column.
- {doc}`temporal_truncation`: bucketing timestamps: truncate to a period, or snap to a period boundary.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
