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
