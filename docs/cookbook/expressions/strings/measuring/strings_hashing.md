# String hashing

Hashes give you a fixed-width key from arbitrary text, which is how you bucket, partition, or pseudonymize without a lookup table. Encodings move bytes through channels that only accept text. Neither is encryption: a hash is one-way, base64 is not secret at all.

The whole script, executed on every test run:

```{literalinclude} ../../../../../examples/expressions/strings_hashing.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_hashing.py
```

## See also

- {doc}`/cookbook/expressions/strings/matching/strings_extraction`: pulling entities and leading fragments out of free text.
- {doc}`/cookbook/expressions/strings/shaping/strings_padding`: fixed-width keys and cleaning stray whitespace.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
