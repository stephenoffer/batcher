# Reducing a list column to one value per row

These are per-row reductions, not group-by aggregates: ``.list.sum()`` sums *within* each row's list and leaves the row count unchanged. That is the difference between "total per basket" and "total across baskets".

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_aggregate.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_aggregate.py
```
