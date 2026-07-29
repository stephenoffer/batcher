# Cheap yes/no questions about the data, and the column-check shorthands

These short-circuit. ``any_match`` stops at the first matching row rather than counting them all, which makes "does this table contain any bad rows?" much cheaper than "how many bad rows does it contain?".

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/meta_predicates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_predicates.py
```
