# Trust

Decide what counts as a valid row, and who may read which rows and columns.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.1em` Data quality
:link: /user-guide/trust/data-quality
:link-type: doc
Expectations, and the fail/drop/quarantine choice.
:::

:::{grid-item-card} {octicon}`law;1.1em` Data contracts
:link: /user-guide/trust/data-contracts
:link-type: doc
Row counts, distributions, freshness, and schema: the checks no single row fails.
:::

:::{grid-item-card} {octicon}`shield-lock;1.1em` Governance and security
:link: /user-guide/trust/governance
:link-type: doc
Column masks, row-level security, lineage, audit.
:::

:::{grid-item-card} {octicon}`file-directory;1.1em` How a table is named
:link: /user-guide/trust/table-names
:link-type: doc
The name a policy is keyed on, and which path spellings fold together.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Write privileges
:link: /user-guide/trust/write-privileges
:link-type: doc
Which privilege each write needs, and the two operations refused inside a policy block.
:::

:::{grid-item-card} {octicon}`key;1.1em` Secrets and keys
:link: /user-guide/trust/secrets
:link-type: doc
Encryption keys and connector credentials, passed by reference.
:::

:::{grid-item-card} {octicon}`lock;1.1em` Hardening a deployment
:link: /user-guide/trust/hardening
:link-type: doc
The settings to change before production, and the boundaries Batcher does not enforce.
:::
::::

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
