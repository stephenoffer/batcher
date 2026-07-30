# Column lineage: which inputs does this output column actually depend on?

Lineage is computed from the plan, so it is exact rather than a guess from parsing SQL text. That is what makes it usable for an impact analysis: if this source column changes, which outputs move?

The whole script, executed on every test run:

```{literalinclude} ../../../examples/governance/lineage.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/governance/lineage.py
```

## See also

- {doc}`masking_and_filters`: column masking and row filtering as a plan rewrite, not a wrapper.
- {doc}`pii_transforms`: masking, hashing, and encrypting a sensitive column.
- {doc}`../../user-guide/governance`: row filters and column masks as a plan rewrite.
- {doc}`../../user-guide/hardening`: the trust boundaries governance does and does not cover.
