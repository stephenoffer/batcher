# String padding and trimming: fixed-width keys and cleaning stray whitespace

Padding matters when a join key is stored at different widths in two systems: an account id written ``42`` in one export and ``000042`` in another will not join until one side is padded. Trimming matters because a trailing space is invisible and breaks equality.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_padding.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_padding.py
```
