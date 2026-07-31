# Governance cookbook

This section holds 3 runnable recipes for restricting and tracing sensitive data, all of it applied as a plan rewrite rather than a wrapper around the result.

That distinction is what makes the enforcement hard to bypass: the restriction is part of the plan the engine runs, so it survives a filter, a join, and a write.

Every page embeds a complete, self-contained script from the [`examples/governance/`](https://github.com/batcher/batcher/tree/main/examples/governance) directory. The scripts build their own in-memory data and assert on their own output, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`masking_and_filters` | Column masking and row filtering as a plan rewrite |
| {doc}`pii_transforms` | Masking, hashing, and encrypting a sensitive column |
| {doc}`lineage` | Which inputs an output column actually depends on |

## See also

- {doc}`/user-guide/trust/governance`: the guide, including grants and the audit trail.
- {doc}`/api/operations/governance`: `SecurityCatalog`, `Principal`, and the policy objects.
- {doc}`/user-guide/trust/secrets`: passing keys and credentials by reference.
- {doc}`/user-guide/trust/hardening`: what to change before production, and what Batcher does not enforce.

```{toctree}
:hidden:

masking_and_filters
pii_transforms
lineage
```
