# Character-class ratios: cheap quality signals for a text corpus

These are the filters that keep junk out of a training set. A row that is 60% digits is probably a table dump; one that is 90% uppercase is probably a shouting header; one with a high non-ASCII ratio may be the wrong language or mojibake. Each ratio is a float in [0, 1] computed in one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_ratios.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_ratios.py
```
