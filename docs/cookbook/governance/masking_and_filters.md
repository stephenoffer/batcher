# Masking and row filters

Batcher enforces governance as a plan rewrite. A `SecurityCatalog` declares the policy, a `Principal` is the identity, and `bt.security(...)` installs both for a block. The policy is compiled into the plan before it runs, so there is no unenforced path around it and no per-row Python check.

The script grants an analyst three of four columns, masks every column tagged `pii`, and limits the analyst to EU rows, while an admin sees everything. Its last check sums a column as the analyst and gets the filtered total. The filter sits inside the plan, so the aggregate never sees the hidden rows.

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

- {doc}`lineage`: which inputs does this output column actually depend on?
- {doc}`pii_transforms`: masking, hashing, and keyed hashing of a sensitive column.
- {doc}`/user-guide/trust/governance`: row filters and column masks as a plan rewrite.
- {doc}`/user-guide/trust/hardening`: the trust boundaries governance does and does not cover.
