# Governance cookbook

Batcher enforces data policy inside the query plan. Grants, column masks, and row filters are compiled into the plan before it runs, so a restricted user's filter, join, aggregate, or write sees only what the policy allows. There is no unenforced path around it.

These three recipes cover the day-to-day work: restricting who sees which rows and columns, protecting a sensitive value while keeping it joinable, and tracing which outputs a source column reaches.

Every page embeds a complete, self-contained script from the [`examples/governance/`](https://github.com/stephenoffer/batcher/tree/main/examples/governance) directory. The scripts build their own in-memory data and assert on their own output, and [`tests/docs/test_examples.py`](https://github.com/stephenoffer/batcher/blob/main/tests/docs/test_examples.py) runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`masking_and_filters` | Column masking and row filtering as a plan rewrite |
| {doc}`pii_transforms` | Masking, hashing, and keyed hashing of a sensitive column |
| {doc}`lineage` | Which inputs an output column actually depends on |

## See also

- {doc}`/user-guide/trust/governance`: the guide, including grants and the audit trail.
- {doc}`/api/operations/governance`: {py:class}`SecurityCatalog <batcher.SecurityCatalog>`, {py:class}`Principal <batcher.Principal>`, and the policy objects.
- {doc}`/user-guide/trust/secrets`: passing keys and credentials by reference.
- {doc}`/user-guide/trust/hardening`: what to change before production, and what Batcher does not enforce.

```{toctree}
:hidden:

masking_and_filters
pii_transforms
lineage
```
