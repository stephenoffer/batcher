# String slicing

Codes, prefixes, and delimited identifiers all need a fixed piece of every value. `left` and `right` take from the ends, `slice` and `substr` take from an offset, and `split_part` takes the nth field of a delimited value.

Check the base before you trust an index. `slice` is 0-based, the Polars spelling. `substr` and `split_part` are 1-based like SQL. The script exercises each one, and every one of them is safe on a value shorter than the requested window: you get what is there rather than an error.

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
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
