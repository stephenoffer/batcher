# Join types, key spellings, and the as-of join for time series

The join type decides what happens to rows with no match, which is where most join bugs live. An inner join silently drops them; a left join keeps them with nulls. Decide which you meant before you write it, then assert the row count.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/joins.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/joins.py
```
