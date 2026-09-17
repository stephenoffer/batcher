# Trust

This section covers the two questions every production table has to answer: is this data right, and who is allowed to see it. Batcher answers both inside the query plan, so your checks and your policies run on the same engine as the query itself.

A data-quality check in Batcher is a boolean expression, and a chain of them lowers to ordinary filters, aggregates, and joins. There is no second validation engine to deploy or keep in sync. You can fail a run, drop the bad rows, quarantine them to a dead-letter sink, or label each row with the rules it broke, all from one chain on `ds.dq`.

Governance works the same way. When a governed table is read, Batcher rewrites the plan so the principal only ever sees the columns it may select, through their masks, and the rows its policy allows. The rewrite happens before the optimizer runs, so `collect`, `count`, `iter_batches`, `write`, and a distributed run all execute the same governed plan. Masks such as `hmac_sha256` and `aes_encrypt` are expressions that run in the Rust data plane, and every decision can be audited.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.1em` Data quality
:link: /user-guide/trust/data-quality
:link-type: doc
Row-level expectations, and the choice between fail, drop, quarantine, and annotate.
:::

:::{grid-item-card} {octicon}`law;1.1em` Data contracts
:link: /user-guide/trust/data-contracts
:link-type: doc
Row counts, distributions, freshness, and schema: the checks no single row fails.
:::

:::{grid-item-card} {octicon}`shield-lock;1.1em` Governance and security
:link: /user-guide/trust/governance
:link-type: doc
Column masks, row-level security, lineage, and audit, enforced as a plan rewrite.
:::

:::{grid-item-card} {octicon}`file-directory;1.1em` How a table is named
:link: /user-guide/trust/table-names
:link-type: doc
The name a policy is keyed on, and which path spellings fold together.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Write privileges
:link: /user-guide/trust/write-privileges
:link-type: doc
`INSERT`, `UPDATE`, and `DELETE` on every write path, and the two rewrites a policy block refuses.
:::

:::{grid-item-card} {octicon}`key;1.1em` Secrets and keys
:link: /user-guide/trust/secrets
:link-type: doc
Keys and connector credentials passed by reference, resolved on the machine that uses them.
:::

:::{grid-item-card} {octicon}`lock;1.1em` Hardening a deployment
:link: /user-guide/trust/hardening
:link-type: doc
Mandatory governance, verified identities, UDF isolation, and admission control.
:::
::::

## Where to start

Start with {doc}`data-quality` if you are writing your first checks, and move to {doc}`data-contracts` when the failure you care about belongs to the whole table rather than a row. Start with {doc}`governance` if you need to restrict who reads what, then read {doc}`hardening` before you deploy, because that page is where governance becomes mandatory rather than opt-in.

```{toctree}
:hidden:

data-quality
data-contracts
governance
table-names
write-privileges
secrets
hardening
```
