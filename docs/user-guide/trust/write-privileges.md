# Write privileges

This page covers the privileges a write needs, how Batcher decides which ones, and the
two operations it refuses inside a `security()` block.

It continues {doc}`Governance and security </user-guide/trust/governance>`, which
introduces the catalog, principals and column access. The examples below assume that
page's setup.

```python
import os
import tempfile

import batcher as bt

d = tempfile.mkdtemp()
customers = os.path.join(d, "customers.parquet")
bt.from_pydict(
    {"id": [1, 2], "email": ["a@x.com", "b@x.com"], "salary": [100, 200]}
).write(customers, format="parquet")
analyst = bt.Principal("ana", roles=["analyst"])
catalog = bt.SecurityCatalog().grant("analyst", on=customers)
```

`SELECT` is not the only privilege. A read policy says nothing about the table a query
*writes*, and leaving that ungoverned undoes the read policy: join the masked table,
write the result somewhere unmentioned, read it back in full.

The privileges are {py:data}`PRIVILEGES <batcher.governance.PRIVILEGES>`, the SQL ones,
spelled the way Snowflake and Unity Catalog spell them: `SELECT`, `INSERT`, `UPDATE`,
`DELETE`.

```python
loading = os.path.join(d, "daily.parquet")

writes = (
    bt.SecurityCatalog()
    .grant("loader", on=loading, privilege="INSERT")
    .grant("owner", on=loading, privilege="INSERT")
    .grant("owner", on=loading, privilege="DELETE")
)
loader = bt.Principal("etl", roles=["loader"])
owner = bt.Principal("ops", roles=["owner"])

with bt.security(writes, loader):
    bt.from_pydict({"day": ["mon"], "n": [1]}).write(loading, format="parquet", mode="error")

print(bt.read.parquet(loading).to_pydict())
```

**Granting one privilege does not confer another**, which is the whole reason to grant
`INSERT` rather than "write". The load job above can add today's data and cannot drop
yesterday's, because `overwrite` destroys the rows already there and therefore needs
`DELETE` as well:

```python
from batcher._internal.errors import AccessDeniedError

with bt.security(writes, loader):
    try:
        bt.from_pydict({"day": ["tue"], "n": [2]}).write(
            loading, format="parquet", mode="overwrite"
        )
    except AccessDeniedError as exc:
        print(exc)
```

The owner holds both and can:

```python
with bt.security(writes, owner):
    bt.from_pydict({"day": ["tue"], "n": [2]}).write(loading, format="parquet", mode="overwrite")

print(bt.read.parquet(loading).to_pydict())
```

Which privileges a write needs comes from what it does to the rows already there, not
from the format:

| Write | Needs |
|---|---|
| `mode="append"`, `"error"`, `"ignore"` | `INSERT` |
| `mode="overwrite"`, `"overwrite_partitions"` | `INSERT`, `DELETE` |
| `mode="upsert"` | `INSERT`, `UPDATE` |
| `mode="update"` | `UPDATE` |
| `mode="delete"` | `DELETE` |
| `ds.merge(...)` | whatever its `WHEN` clauses do |
| A streaming write | `INSERT`, checked once before the query starts |

A `MERGE` asks only for what its clauses actually do, so an insert-only upsert needs
`INSERT` alone. A privilege nobody can grant narrowly is one every role ends up holding.

Every write path is covered, batch and streaming, single-node and distributed, including a
native `MERGE INTO` a Delta or Iceberg table where the format's own client does the work. A
streaming write is authorized before the query starts rather than at the first micro-batch,
because a stream refused after its first batch has already written.

```{important}
**One grant governs the whole table.** A table nobody has granted anything on is open, as
before. Once any grant names it, every privilege on it is deny-by-default. So a catalog
that grants `SELECT` and nothing else makes that table readable by the granted roles and
writable by nobody. If a pipeline reads and rewrites the same governed table, grant it the
write privileges explicitly.
```

Write decisions are audited in the same {py:class}`GovernanceEvent <batcher.GovernanceEvent>`
shape a read produces, with `privilege` naming the write and `visible` naming the columns
written, so "who touched this table" is one query over one log.

## Maintenance runs outside the block

{py:func}`bt.compact() <batcher.compact>` reads a table and writes the result back over it.
Inside a `security()` block that read is the *principal's* view, so the write would replace
the table with it: the masked value in place of the real one, and no column at all where the
principal had no `SELECT`. Batcher refuses it rather than doing it.

```python
from batcher._internal.errors import AccessDeniedError

try:
    with bt.security(catalog, analyst):
        bt.compact(customers, format="parquet")
except AccessDeniedError as exc:
    print(exc)
```

Run maintenance outside the block, with the engine's own authority over the table, which is
how a warehouse runs `OPTIMIZE` anyway.

{py:func}`bt.vacuum() <batcher.vacuum>` is different: it deletes files no live version
references, so it never writes a view back. A real vacuum needs `DELETE`. A dry run needs
nothing, because it deletes nothing and it is the check you run *before* deciding whether
the deletion is safe.
