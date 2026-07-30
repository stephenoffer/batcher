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

## See also

- {doc}`structs_and_maps`: struct and map columns: nested records without flattening the table.
- {doc}`temporal_differences`: durations between two timestamp columns, and shifting a timestamp.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
