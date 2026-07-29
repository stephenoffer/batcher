# Masking, hashing, and encrypting a sensitive column

These are ordinary expressions, so they run in Rust at full speed and compose with everything else. Pick by what you need back: masking is one-way and readable, hashing is one-way and joinable, encryption is reversible with the key.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/governance/pii_transforms.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/governance/pii_transforms.py
```
