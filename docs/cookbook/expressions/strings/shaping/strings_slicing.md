# String slicing

``head``/``tail`` take from the ends, ``slice``/``substr`` take from an offset, and ``split_part`` takes the nth field of a delimited value. All of them are safe on values shorter than the requested window: you get what is there rather than an error.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_slicing.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_slicing.py
```

## See also

- {doc}`/cookbook/expressions/strings/matching/strings_similarity`: fuzzy string matching against a reference value.
- {doc}`/cookbook/expressions/nested/structs_and_maps`: nested records without flattening the table.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
