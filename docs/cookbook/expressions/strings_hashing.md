# Hashing and encoding a string column: keys, checksums, and safe transport

Hashes give you a fixed-width key from arbitrary text, which is how you bucket, partition, or pseudonymize without a lookup table. Encodings move bytes through channels that only accept text. Neither is encryption: a hash is one-way, base64 is not secret at all.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/strings_hashing.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/strings_hashing.py
```

## See also

- {doc}`strings_extraction`: pulling entities and leading fragments out of free text.
- {doc}`strings_padding`: string padding and trimming: fixed-width keys and cleaning stray whitespace.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
