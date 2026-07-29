# Weekend and business-day predicates, and formatting a timestamp for output

Reports almost always want weekdays only, and almost always want a string at the end. Both are expressions, so the filter pushes down toward the scan and the formatting happens in Rust rather than in a Python ``strftime`` loop.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/temporal_business_days.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_business_days.py
```
