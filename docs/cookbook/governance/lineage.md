# Column lineage

Lineage is computed from the plan rather than guessed by parsing SQL text, so it is exact. Exactness is what an impact analysis needs: if this source column changes, which outputs move?

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
- {doc}`/user-guide/trust/governance`: row filters and column masks as a plan rewrite.
- {doc}`/user-guide/trust/hardening`: the trust boundaries governance does and does not cover.
