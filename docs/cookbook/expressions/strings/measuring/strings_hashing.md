# String hashing

A hash gives you a fixed-width key from arbitrary text, which is how you bucket, partition, or pseudonymize without a lookup table. An encoding moves bytes through a channel that only accepts text. Neither is encryption: a hash is one-way, and base64 is not secret at all.

The script hashes an email column with `hash64`, `xxhash64`, `crc32`, `md5`, `sha1`, and `sha256`, shows that a digest is deterministic, and derives a stable shard number from the key. It round-trips values through base64, hex, and URL encoding. It also shows byte length and character length diverging once the text leaves ASCII.

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
