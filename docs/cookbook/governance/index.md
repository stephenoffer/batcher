# Governance cookbook

Column masking, row filters, PII transforms, and column lineage, all applied as plan rewrites.

Every page here embeds a complete, self-contained script from the
[`examples/governance/`](https://github.com/batcher/batcher/tree/main/examples/governance) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`lineage` | Column lineage: which inputs does this output column actually depend on? |
| {doc}`masking_and_filters` | Column masking and row filtering as a plan rewrite, not a wrapper |
| {doc}`pii_transforms` | Masking, hashing, and encrypting a sensitive column |

```{toctree}
:hidden:

lineage
masking_and_filters
pii_transforms
```
