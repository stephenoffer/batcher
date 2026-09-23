# PII transforms

Masking and hashing are ordinary expressions in Batcher. They run in Rust over whole columns and compose with everything else, so protecting a column is one `with_columns` call.

Pick by what you need back. {py:obj}`bt.mask <batcher.mask>` keeps a readable tail for a human to recognize the record. `.str.sha256()` is one-way but deterministic, so the script joins two tables on the hashed email. {py:obj}`bt.hmac_sha256 <batcher.hmac_sha256>` adds a key, so the same value hashes differently in another system. The key is passed as an `env:` reference that resolves at execution time, and the script asserts the secret never appears in `explain()` output.

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
