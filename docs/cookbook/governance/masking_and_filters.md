# Column masking and row filtering as a plan rewrite, not a wrapper

Governance in Batcher is a *rewrite*: the policy is compiled into the plan before it runs, so there is no unenforced path around it and no per-row Python check. A ``SecurityCatalog`` declares the policy, a ``Principal`` is the identity, and ``bt.security(...)`` installs both for the duration of a block.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/governance/masking_and_filters.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/governance/masking_and_filters.py
```

## See also

- {doc}`lineage`: column lineage: which inputs does this output column actually depend on?
- {doc}`pii_transforms`: masking, hashing, and encrypting a sensitive column.
- {doc}`../../user-guide/governance`: row filters and column masks as a plan rewrite.
- {doc}`../../user-guide/hardening`: the trust boundaries governance does and does not cover.
