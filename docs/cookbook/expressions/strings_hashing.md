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
