# Column lineage

`ds.lineage()` reports which source columns each output column was computed from. Batcher reads that off the optimized plan rather than parsing SQL text, so the answer is exact, and nothing executes to produce it.

Exactness is what an impact analysis needs: if a source column changes, you know which outputs move. The script checks that a derived `revenue` column depends on `price` and `qty` and nothing else, then shows the dependency surviving a `group_by` and an aggregate, which is where tracking lineage by hand usually breaks down.

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
- {doc}`pii_transforms`: masking, hashing, and keyed hashing of a sensitive column.
- {doc}`/user-guide/trust/governance`: row filters and column masks as a plan rewrite.
- {doc}`/user-guide/trust/hardening`: the trust boundaries governance does and does not cover.
