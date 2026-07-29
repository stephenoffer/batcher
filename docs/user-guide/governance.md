# Governance and security

Batcher enforces *who may read which rows and columns, and through what mask*, in the
engine itself. A `SecurityCatalog` declares the policy, a `Principal` is the identity a
query runs as, and `bt.security(...)` binds the two to a scope. Everything read inside
that scope is governed.

Policy is applied when a table is **read**, not when a query is executed. That is what
makes it unbypassable: a `Dataset` never holds an ungoverned plan, so there is no
`count()`, `write()`, or streaming path that could skip the check. A database works the
same way. A masking policy resolves against the role in effect when the column is read.

## Setup

Governance keys on the path a table is read from. That is the only name a file-backed
table has *before* anyone reads it, and a policy has to be declarable before the first
read.

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
that table to deny-by-default**, so installing a catalog never silently locks out
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

Each call records a `Grant`: a role, a table, and the columns that role may read
(`select=None` means all of them). A catalog is a list of those. That is what keeps it a
value you can print and diff and review, rather than a service you have to interrogate.

A column the principal may not select does not exist for it at all. Referencing
`salary` raises the ordinary unknown-column error, not "access denied". An error that
said "you may not read `salary`" would itself confirm that `salary` exists.

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

The two spellings record two different policy objects, and the difference is the whole
reason to classify a column at all. `mask_column` records a `ColumnMask`, bound to one
`table`.`column`: precise, and one entry per column per table. `mask_tag` records a
`TagMask`, bound to a *tag*, so it governs every column carrying that tag in every table,
including the table someone adds next quarter that nobody will remember to come back and
mask. Per-column bindings are what you reach for to override a case; tags are what make
the catalog survive the tenth table.

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

Each call records a `RowFilter`. Note the shape of the predicate: it takes the
*principal*, not a row. It is called once, while the plan is being built, and returns an
ordinary `Expr`, here `col("region") == "EU"`. The engine then treats that expression
like any predicate you wrote yourself: Kyber pushes it down toward the scan and fuses it
with neighboring filters. Row-level security therefore costs one filter, not a policy
callback per row.

## Enforcement is a plan rewrite

`enforce` rewrites the plan when a governed table is read. Every governed scan becomes:

```text
Project(visible columns, read through their masks)
  Filter(row-access predicate)
    Scan(table)
```

You can see it:

```python
with bt.security(catalog, analyst):
    print(bt.read.parquet(customers).explain())
```

The `project` is the columns the analyst may select, read through their masks; the
`filter` under it is the row-access predicate; the `scan` sees the whole table. The
estimate drops from 4 rows to 2 because the row policy is a predicate like any other:

```text
project                         est≈2 (learned)
  filter                        est≈2 (learned)
    scan                        est≈4 (exact)

decisions:
  - [core/io] source read at 0 MB/s (learned) — ~0.0s to read
```

(The throughput under `decisions:` is measured; it reads 0 MB/s here because the whole
table is a few kilobytes.)

The order matters in both directions. The filter sits *below* the projection, so a row
policy may reference `region` even though the analyst has no `SELECT` on it; a row-access
policy runs with the catalog's authority, not the caller's. The masked projection sits at
the leaf, below everything the user wrote, so no operator above it can ever see a raw
value.

Because it is a rewrite and not a check, there is no enforcement code on the hot path to
bypass and no execution mode that skips it. `collect`, `count`, `iter_batches`, `write`,
and the distributed path all run the same governed plan.

## Choosing a protection

| Function        | Reversible?       | Joinable? | Use it for                   |
| --------------- | ----------------- | --------- | ---------------------------- |
| `mask`          | no                | no        | partial disclosure to humans |
| `hmac_sha256`   | no                | yes       | pseudonymized analytics      |
| `aes_encrypt`   | yes, with the key | yes       | data that must be read back  |

"Joinable" means equal inputs produce equal outputs, so the protected column still
groups and equi-joins. Every expression in Batcher must be deterministic (the sequential
interpreter is the correctness oracle that the parallel executor and the JIT are checked
against), so `aes_encrypt` uses AES-256-GCM-SIV, the AEAD whose security does not
collapse under the fixed nonce that determinism forces.

The price of that determinism is that equality is observable: an encrypted column
reveals which rows share a value. Where that is unacceptable, do not protect the column.
Leave it out of the projection.

```python
key = "00" * 32  # 32 bytes as hex; use a secret manager in production

enc = bt.from_pydict({"ssn": ["123-45-6789"]}).select(c=bt.aes_encrypt(bt.col("ssn"), key))
print(enc.select(s=bt.aes_decrypt(bt.col("c"), key)).to_pydict())
```

`aes_decrypt` under the wrong key yields NULL rather than failing the query. One
unreadable row must not abort a scan of a billion. An all-NULL result is the unambiguous
signal that the key is wrong.

For pseudonymized analytics, prefer `hmac_sha256` over a bare `sha256`: a plain digest
of a low-entropy value such as an email address is recovered by trying every email
address, while an HMAC is not, because the attacker lacks the key.

```python
users = bt.from_pydict({"email": ["a@x.com", "a@x.com", "b@x.com"]})
out = users.select(p=bt.hmac_sha256(bt.col("email"), key="s3cret")).to_pydict()
print(out["p"][0] == out["p"][1] != out["p"][2])  # stable, so it still joins
```

Where the key itself comes from — an environment variable, a mounted file, or Vault via a
helper command — is covered in {doc}`secrets`, along with the same indirection for
connector passwords and API tokens.

## Fingerprints and change detection

Hiding a value is half of governance. The other half is proving what a value *was*,
without keeping the plaintext around. A hash reduces a row (or a column) to a fixed
digest you can compare, deduplicate, or diff across snapshots.

`hash_rows(*exprs, seed=0)` is a deterministic 64-bit digest of several columns at once,
stable across partitions and runs and machines. It is the change-detection and dedup-key
primitive: fingerprint each row, and any later row whose fingerprint differs is one that
changed.

```python
cols = [bt.col("id"), bt.col("email"), bt.col("region")]
before = bt.from_pydict(
    {"id": [1, 2, 3], "email": ["a@x.com", "b@x.com", "c@x.com"], "region": ["EU", "EU", "US"]}
)
after = bt.from_pydict(
    {"id": [1, 2, 3], "email": ["a@x.com", "b@new.com", "c@x.com"], "region": ["EU", "EU", "US"]}
)

fp_before = before.select(fp=bt.hash_rows(*cols)).to_pydict()["fp"]
fp_after = after.select(fp=bt.hash_rows(*cols)).to_pydict()["fp"]
print([i for i, (a, b) in enumerate(zip(fp_before, fp_after)) if a != b])  # [1] — only row 1 changed
```

For a single column, the `.str` digests produce a per-value fingerprint.
`.str.xxhash64()` and `.str.hash64()` are fast non-cryptographic 64-bit hashes, the ones
to reach for when sharding or bucketing.

```python
accounts = bt.from_pydict({"email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"]})
print(accounts.select(bucket=bt.col("email").str.xxhash64().abs() % 4).to_pydict()["bucket"])
# [3, 3, 2, 0]
```

`.str.md5()` and `.str.sha1()` return the lowercase-hex cryptographic digest
(matching DuckDB's `md5` / `sha1`), and `.str.crc32()` returns the CRC-32 integrity
checksum:

```python
sample = bt.from_pydict({"s": ["abc"]})
print(sample.select(m=bt.col("s").str.md5(), c=bt.col("s").str.crc32()).to_pydict())
# {'m': ['900150983cd24fb0d6963f7d28e17f72'], 'c': [891568578]}
```

The same caution the table draws for a bare `sha256` applies to these: a `md5` or
`sha1` digest is **not** a safe pseudonym. Both are cryptographic hashes, but a digest of
a low-entropy value such as an email address is recovered by hashing every candidate. It
leaks exactly what a pseudonym must hide. For pseudonymized-but-joinable analytics use
`hmac_sha256` (above), whose key the attacker lacks. Reserve `hash_rows`, `md5`, `sha1`,
`crc32`, and the `xxhash64`/`hash64` pair for fingerprinting, bucketing, and integrity
checks, never for de-identification.

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
ungoverned, and an in-memory dataset (`from_pydict`, `from_arrow`) is never governed:
there is no durable name to write a policy about a dict you are already holding.

## Auditing

Every governed read emits a `GovernanceEvent`: who asked, what they were allowed to see,
what was withheld, what was masked, which row filters applied. Denials are audited too,
before the error is raised. The access a compliance review most wants to find is the one
that was refused.

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
the plan, executing nothing, and reports the source columns each output column's *values*
are derived from.

```python
derived = bt.read.parquet(customers).select(
    key=bt.hmac_sha256(bt.col("email"), key="s3cret"),
    region=bt.col("region"),
)

pii = f"{customers}.email"
carrying = [col for col, origins in derived.lineage().items() if pii in origins]
print(carrying)
```

Lineage is what makes a tag durable. A masked value that has been renamed, cast,
concatenated, or aggregated is no longer called `email` and no longer lives in
`customers`. It still carries what `customers.email` held, and lineage is how you find
it. Tag the source column once; ask lineage which output columns descend from it.

`ds.lineage()` renders origins as `"table.column"` strings. Underneath it is
`column_lineage(plan, tables)`, which works on a `LogicalPlan` and returns each column's
origins as a set of `Origin`, the `(table, column)` pair. Reach for it when you are
building policy tooling of your own and want the pair rather than the rendered string.

```python
import pyarrow as pa

from batcher.governance import column_lineage
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Project, Projection, Scan
from batcher.plan.schema import SchemaRef

schema = SchemaRef.from_arrow(pa.schema([("first", pa.string()), ("last", pa.string())]))
plan = Project(Scan(0, schema), (Projection(alias="name", expr=Col("first") + Col("last")),))

print(sorted(column_lineage(plan, ["people.parquet"])["name"]))
# [('people.parquet', 'first'), ('people.parquet', 'last')]
```

Two properties make the answer trustworthy:

* **Data flow, not control flow.** Filtering on `email` does not put it in the lineage of
  the surviving columns; its values never reach the output. That matches how Unity
  Catalog and Snowflake report lineage.
* **Over-approximate, never under-approximate.** An opaque `map_batches` stage is treated
  as though every output column derives from every input column. A false "this might carry
  PII" costs a review; a false "it cannot" costs a breach.

A column built only from literals, or generated by `with_row_index`, has no origin.

## Persisting a policy

The masks and row filters above are Python callables. Flexible in process, impossible to
store. For a policy your platform keeps in an external store and reconstructs each
session, build the catalog from the **declarative** factories instead. They are picklable,
so a catalog built from them survives a round-trip and enforces identically.

```python
from batcher.governance import Pseudonymize, Nullify, MatchesAttribute

catalog = (
    bt.SecurityCatalog()
    .grant("analyst", on=customers, select=["id", "email", "region"])
    .mask_tag("pii", Pseudonymize("env:PII_KEY"))          # keyed pseudonym
    .mask_column(customers, "salary", Nullify())            # full redaction
    .filter_rows(customers, MatchesAttribute("region", "region"))
)
```

`Redact`, `Pseudonymize`, `Encrypt`, and `Nullify` cover the masking shapes an enterprise
policy uses (and lower to the data-plane functions above); `MatchesAttribute` and
`AttributeIn` cover attribute-based row access. Batcher persists nothing itself. It hands
you picklable policy objects and enforces the catalog you give it. Where that policy lives
is your platform's decision.

## What this does not do

Batcher authorizes; it does not authenticate, and it does not encrypt data at rest or
in transit for you. It has no persistent policy store either. A `SecurityCatalog` is
built in Python, which makes it a serializable, diffable artifact you can load from your
own store and check into review.

## See also

- [Data quality](data-quality.md): validate and quarantine rows before they reach a
  consumer.
- [Complete API reference](../api/complete.md): `SecurityCatalog`, `Principal`,
  `GovernanceEvent`, and `security`.
- [Agent skills](../agents/index.md): `apply-governance-and-security`, the same
  surface as a procedure, with what to verify before trusting an enforced plan.
