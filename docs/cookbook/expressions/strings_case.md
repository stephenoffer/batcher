# String case: normalizing capitalization before you compare or group

Case folding is the cheapest way to stop a group_by splitting "ACME", "Acme", and "acme" into three groups. Do it once in the projection, then group on the normalized column.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_case.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_case.py
```
