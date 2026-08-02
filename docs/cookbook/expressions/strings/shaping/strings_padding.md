# Padding and trimming

Padding matters when a join key is stored at different widths in two systems: an account id written ``42`` in one export and ``000042`` in another will not join until one side is padded. Trimming matters because a trailing space is invisible and breaks equality.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_padding.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_padding.py
```

## See also

- {doc}`/cookbook/expressions/strings/measuring/strings_hashing`: keys, checksums, and safe transport.
- {doc}`/cookbook/expressions/strings/measuring/strings_paths`: parsing file paths held in a column.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
