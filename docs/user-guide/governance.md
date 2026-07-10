# Governance and security

Batcher enforces *who may read which rows and columns, and through what mask*, in the
engine itself. A `SecurityCatalog` declares the policy, a `Principal` is the identity a
query runs as, and `bt.security(...)` binds the two to a scope. Everything read inside
that scope is governed.

Policy is applied when a table is **read**, not when a query is executed. That is what
makes it unbypassable: a `Dataset` never holds an ungoverned plan, so there is no
`count()`, `write()`, or streaming path that could skip the check. It is also what a
database does — a masking policy resolves against the role in effect when the column is
read.

## Setup

Governance keys on the path a table is read from, because that is the only name a
file-backed table has *before* anyone reads it — and a policy has to be declarable
before the first read.

```python
import os
import tempfile

import batcher as bt

d = tempfile.mkdtemp()
customers = os.path.join(d, "customers.parquet")

bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "email": ["a@x.com", "b@x.com", "c@y.com", "d@y.com"],
        "region": ["EU", "EU", "US", "US"],
        "salary": [100, 200, 300, 400],
    }
).write(customers, format="parquet")
```

## Principals

A principal has a name, roles, and free-form attributes. Attributes are what let one
row-access policy serve every regional analyst instead of one policy per region.

```python
analyst = bt.Principal("ana", roles=["analyst"], attrs={"region": "EU"})
admin = bt.Principal("root", roles=["admin"], attrs={"region": "EU"})
intern = bt.Principal("ivy", roles=["intern"])
```

A principal carries no credentials. Authentication happens outside the engine; Batcher
only authorizes.

## Column access

`grant` gives a role `SELECT` on some columns. The **first grant on a table switches
that table to deny-by-default** — so installing a catalog never silently locks out
queries against tables nobody wrote a policy about.

```python
catalog = (
    bt.SecurityCatalog()
    .grant("analyst", on=customers, select=["id", "email", "region"])
    .grant("admin", on=customers)  # no `select` → every column
)

with bt.security(catalog, analyst):
    ds = bt.read.parquet(customers)

print(ds.columns)
```

A column the principal may not select simply does not exist for it. Referencing
`salary` raises the ordinary unknown-column error rather than "access denied" — an
error that said "you may not read `salary`" would itself confirm that `salary` exists.

## Masking PII

Classify a column once with `tag`, then govern every column carrying that tag with
`mask_tag`. This is what keeps a catalog manageable past a handful of tables: tag
`email` as `pii` wherever it appears, write one policy, and tables added later are
covered automatically.

The masks themselves are ordinary expressions ([`mask`](../api/complete.md),
`hmac_sha256`, `aes_encrypt`), so they run in the Rust data plane at full speed.

```python
catalog = (
    bt.SecurityCatalog()
    .grant("analyst", on=customers, select=["id", "email", "region"])
    .grant("admin", on=customers)
    .tag(customers, "email", "pii")
    .mask_tag("pii", lambda c: bt.mask(c, show_last=6), exempt=["admin"])
)

with bt.security(catalog, analyst):
    print(bt.read.parquet(customers).sort("id").to_pydict())
```

Masking is applied **at the scan**, below anything the user wrote, so the raw value
never exists anywhere in the plan. A principal cannot recover it by filtering on the
column, grouping by it, or joining it against known plaintext:

```python
with bt.security(catalog, analyst):
    ds = bt.read.parquet(customers)

print(ds.filter(bt.col("email") == "a@x.com").count())  # 0 — the filter sees the mask
```

An explicit `mask_column` overrides the tag-derived mask for that one column, and a
principal holding an `exempt` role reads the raw value.

## Row-level security

`filter_rows` restricts a table to the rows a principal may see. The predicate is a
function of the principal, so it can compare against its attributes.

```python
catalog = catalog.filter_rows(
    customers,
    lambda p: bt.col("region") == p.attrs["region"],
    exempt=["admin"],
)

with bt.security(catalog, analyst):
    print(bt.read.parquet(customers).count())  # 2 — only the EU rows
```

The row filter is applied **below** column pruning, so a policy may reference columns
the principal cannot itself select. That is what lets the filter above work for an
analyst with no `SELECT` on `region`. A row-access policy runs with the catalog's
authority, not the caller's.

Multiple row filters on one table are conjoined, so adding one can never widen what a
principal sees.

## Choosing a protection

| Function        | Reversible?       | Joinable? | Use it for                   |
| --------------- | ----------------- | --------- | ---------------------------- |
| `mask`          | no                | no        | partial disclosure to humans |
| `hmac_sha256`   | no                | yes       | pseudonymized analytics      |
| `aes_encrypt`   | yes, with the key | yes       | data that must be read back  |

"Joinable" means equal inputs produce equal outputs, so the protected column still
groups and equi-joins. Every expression in Batcher must be deterministic — the
sequential interpreter is the correctness oracle the parallel executor and the JIT are
checked against — so `aes_encrypt` uses AES-256-GCM-SIV, the AEAD whose security does
not collapse under the fixed nonce that determinism forces.

The price of that determinism is that equality is observable: an encrypted column
reveals which rows share a value. Where that is unacceptable, do not protect the
column — leave it out of the projection.

```python
key = "00" * 32  # 32 bytes as hex; use a secret manager in production

enc = bt.from_pydict({"ssn": ["123-45-6789"]}).select(c=bt.aes_encrypt(bt.col("ssn"), key))
print(enc.select(s=bt.aes_decrypt(bt.col("c"), key)).to_pydict())
```

`aes_decrypt` under the wrong key yields NULL rather than failing the query — one
unreadable row must not abort a scan of a billion. An all-NULL result is the
unambiguous signal that the key is wrong.

For pseudonymized analytics, prefer `hmac_sha256` over a bare `sha256`: a plain digest
of a low-entropy value like an email address is recovered by trying every email
address, while an HMAC is not, because the attacker lacks the key.

```python
users = bt.from_pydict({"email": ["a@x.com", "a@x.com", "b@x.com"]})
out = users.select(p=bt.hmac_sha256(bt.col("email"), key="s3cret")).to_pydict()
print(out["p"][0] == out["p"][1] != out["p"][2])  # stable, so it still joins
```

:::{warning}
A key passed to `aes_encrypt` / `aes_decrypt` / `hmac_sha256` is embedded in the query's
JSON plan IR, which crosses the FFI boundary and may be captured by plan logging or the
profile log. Source keys from a secret manager and treat a serialized plan as secret
material. An expression's `repr` redacts the key, so tracebacks and notebook echoes do
not leak it.
:::

## Scoping

`bt.security(...)` is a context manager, not a setter, so a read cannot precede the
policy that governs it. A `Dataset` built inside a block keeps that block's policy for
its whole life, including terminal operations run after the block exits.

```python
with bt.security(catalog, analyst):
    ds = bt.read.parquet(customers)

print(ds.count())  # still 2 — the plan was governed when the table was read
```

Nested blocks restore the outer policy on exit. A table read outside any block is
ungoverned, and an in-memory dataset (`from_pydict`, `from_arrow`) is never governed —
there is no durable name to write a policy about a dict you are already holding.

## Auditing

Every governed read emits a `GovernanceEvent` — who asked, what they were allowed to
see, what was withheld, what was masked, and which row filters applied. Denials are
audited too, before the error is raised: the access a compliance review most wants to
find is the one that was refused.

```python
seen = []

with bt.security(catalog, analyst, audit=seen.append):
    bt.read.parquet(customers)

print(seen[0].visible, seen[0].denied, seen[0].masked)
```

The event names columns and policies, never **values** and never key material, so it is
safe to write to a log that outlives the data. It is produced by the same traversal that
rewrites the plan, so what you record is by construction what was enforced. Every
decision is also logged at `INFO` on the `batcher.governance` logger, whether or not you
pass a sink.

## Column-level lineage

Tagging `email` as PII only helps if you can answer where it went. `ds.lineage()` reads
the plan — nothing executes — and reports, per output column, the source columns its
*values* are derived from.

```python
derived = bt.read.parquet(customers).select(
    key=bt.hmac_sha256(bt.col("email"), key="s3cret"),
    region=bt.col("region"),
)

pii = f"{customers}.email"
carrying = [col for col, origins in derived.lineage().items() if pii in origins]
print(carrying)
```

Two properties make the answer trustworthy:

* **Data flow, not control flow.** Filtering on `email` does not put it in the lineage of
  the surviving columns — its values never reach the output. That matches how Unity
  Catalog and Snowflake report lineage.
* **Over-approximate, never under-approximate.** An opaque `map_batches` stage is treated
  as though every output column derives from every input column. A false "this might carry
  PII" costs a review; a false "it cannot" costs a breach.

A column built only from literals, or generated by `with_row_index`, has no origin.

## What this does not do

Batcher authorizes; it does not authenticate, and it does not encrypt data at rest or
in transit for you. It has no persistent policy store — a `SecurityCatalog` is built in
Python, which means it is a serializable, diffable, reviewable artifact you can load
from your own store and check into review.
