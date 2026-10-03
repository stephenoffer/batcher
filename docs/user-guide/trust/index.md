# Trust

This section covers two questions every production table has to answer. Is the data right? And who may see it? Batcher answers both inside the query plan, on the same engine that runs the query.

A data-quality check is a boolean expression. Chain a few and they lower to ordinary filters, aggregates and joins, so there's no second validation engine to deploy. One chain on `ds.dq` can fail the run, drop bad rows, quarantine them, or label each row with the rules it broke:

```python
import os
import tempfile

import batcher as bt

people = bt.from_pydict({"id": [1, 2, 3], "email": ["a@x.io", None, "c@x.io"], "age": [34, 28, 200]})
print(str(people.dq.not_null("email").in_range("age", 0, 120).validate()))
# ValidationReport(violations: not_null(email)=1, in_range(age, 0, 120)=1)

good, bad = people.dq.in_range("age", 0, 120).quarantine()
print(good.to_pydict()["id"], bad.to_pydict()["id"])
# [1, 2] [3]
```

Governance is a plan rewrite too. Read a governed table and Batcher rewrites the plan before anything runs, so the principal sees only the columns it may select, through their masks, and only the rows its policy allows. `collect` gets that plan. So do `write` and a distributed run.

```python
path = os.path.join(tempfile.mkdtemp(), "people.parquet")
people.write(path, format="parquet")

catalog = (
    bt.SecurityCatalog()
    .grant("analyst", on=path, select=["id", "email"])
    .tag(path, "email", "pii")
    .mask_tag("pii", lambda c: bt.mask(c, show_last=4))
)
with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
    print(bt.read.parquet(path).sort("id").to_pydict())
# {'id': [1, 2, 3], 'email': ['XXx.io', None, 'XXx.io']}
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.1em` Data quality
:link: /user-guide/trust/data-quality
:link-type: doc
Row-level expectations, and what to do with the rows that break them.
:::

:::{grid-item-card} {octicon}`law;1.1em` Data contracts
:link: /user-guide/trust/data-contracts
:link-type: doc
Checks no single row can fail: row counts, distributions, freshness, schema.
:::

:::{grid-item-card} {octicon}`shield-lock;1.1em` Governance and security
:link: /user-guide/trust/governance
:link-type: doc
Column masks and row-level security, with lineage and audit, enforced as a plan rewrite.
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
Mandatory governance and verified identities, plus UDF isolation and admission control.
:::
::::

## Where to start

Writing your first checks? Start with {doc}`data-quality`, then move to {doc}`data-contracts` once the failure you care about belongs to the whole table. Restricting who reads what? Start with {doc}`governance`. Read {doc}`hardening` before you deploy, because that's where governance stops being opt-in.

## See also

The recipes and reference behind these guides.

- {doc}`/examples/data-quality`: quality and governance as standalone scripts, each run on every commit.
- {doc}`/cookbook/governance/index`: masking and PII recipes, with lineage.
- {doc}`/api/operations/governance`: the policy types, the enforcement model, and data residency.
- {doc}`/api/symbols/dataset-accessors`: the `ds.dq` and `ds.scd` surfaces these guides use.
- {doc}`/architecture/deep-dives/query/plan-ir`: the plan tree a policy rewrite acts on.

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
