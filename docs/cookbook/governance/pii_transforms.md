# PII transforms

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

## See also

- {doc}`masking_and_filters`: column masking and row filtering as a plan rewrite, not a wrapper.
- {doc}`lineage`: which inputs does this output column actually depend on?
- {doc}`/user-guide/trust/governance`: row filters and column masks as a plan rewrite.
- {doc}`/user-guide/trust/hardening`: the trust boundaries governance does and does not cover.
