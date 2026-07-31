---
name: apply-governance-and-security
description: Apply column masking, row-level security, PII protection, auditing, and column-level lineage to Batcher pipelines using SecurityCatalog, Principal, and the bt.security block — including why enforcement is a plan rewrite and what to verify before trusting it. Invoke when a pipeline must restrict who reads which rows or columns, protect PII, produce a compliance audit trail, or trace where a sensitive column flows.
---

# Apply governance and security

Batcher **authorizes**; it does not authenticate, encrypt at rest, or store policy.
You declare a policy as a `SecurityCatalog`, name the identity as a `Principal`, and
read inside a `bt.security(...)` block. Everything else follows from one fact:

## Enforcement is a plan rewrite

This is the load-bearing concept. When a governed table is read, `enforce` rewrites the
plan — before the optimizer runs — so every governed scan becomes:

```text
Project(visible columns, read through their masks)   <- what the principal may read
  Filter(row-access predicate)                       <- which rows it may see
    Scan(table)                                      <- the raw table
```

Three consequences you must reason from:

1. **Masks are applied at the leaf, below everything the user wrote.** The raw value
   never exists above the scan, so a principal cannot recover it by filtering on it,
   grouping by it, or joining against it. Post-hoc masking (masking a result *after*
   collecting) gives none of this.
2. **The row filter sits below the projection**, so a policy may reference columns the
   principal cannot itself select. A policy runs with the catalog's authority, not the
   caller's. Multiple filters on a table are conjoined — adding one can never widen what
   is visible.
3. **The rewrite composes with the optimizer.** A row filter is an ordinary `Expr`
   returned once at plan-build time, so Kyber pushes it toward the scan and fuses it with
   neighboring filters. Row-level security costs one predicate, not a callback per row —
   and masking runs in the Rust data plane at full speed.

A table no policy mentions is left structurally untouched (`catalog.governs(table)`), so
installing a catalog cannot perturb unrelated queries. A column the principal may not
select is *removed from the scan's output*; a later reference to it fails as an unknown
column — fail-closed, and it does not confirm the column exists. Losing access to *every*
column raises `AccessDeniedError`.

## The shape

```python
import batcher as bt
from batcher.governance import MatchesAttribute, Redact

catalog = (
    bt.SecurityCatalog()
    .grant("analyst", on=customers, select=["id", "email", "region", "amount"])
    .tag(customers, "email", "pii")
    .mask_tag("pii", Redact(show_last=5), exempt=["security"])
    .filter_rows(customers, MatchesAttribute("region", "region"), name="own_region")
)
analyst = bt.Principal("ana", roles=["analyst"], attrs={"region": "EU"})

with bt.security(catalog, analyst):
    ds = bt.read.parquet(customers)      # policy binds at READ time

print(ds.sort("id").to_pydict())
# {'id': [1, 2], 'email': ['XXXXx.com', 'XXXx.com'], 'region': ['EU', 'EU'], 'amount': [10, 20]}
```

`bt.security(catalog, principal, *, audit=None)` is a context manager, not a setter,
because governance is applied when a table is **read**. A `Dataset` created inside the
block keeps its policy for life, including terminal operations run after the block exits.
A table read *outside* any block is ungoverned — that is the single most common mistake.
Tables are named by the path they are read from: the only identity stable across runs and
knowable *before* the read, and a policy must be declarable before the first read.

`bt.Principal(name, roles=..., attrs=...)` is the identity — immutable, carrying no
credentials (authentication happens outside the engine). `roles` drive grants and
exemptions (`has_role`, `has_any_role`); `attrs` drive attribute-based row filters, so one
policy serves every regional analyst. `has_any_role([])` is False — an empty exemption
list exempts nobody, never everybody.

## Column access and masking

- `grant(role, on=table, select=[...])` — a table with **no** grant is open; the first
  grant makes it deny-by-default, and a principal sees the union of its roles' grants
  (`select=None` grants every column).
- `mask_column(table, column, mask, *, exempt=())` — an explicit mask on one column.
- `tag(table, column, *tags)` then `mask_tag(tag, mask, *, exempt=())` — classify once,
  attach policy once, every tagged column in every table covered. Prefer this: it scales,
  and it pairs with lineage.
- `mask_for`, `visible_columns`, `row_filters_for`, `governs` — the resolution side, for
  asserting a policy in a test without running a query.

**Resolution is most-restrictive-wins.** An explicit `ColumnMask` applies if the principal
is not exempt from it; otherwise resolution falls through to the tag masks (first tag in
*sorted* order whose mask applies, so it is deterministic regardless of declaration
order). Being exempt from one policy does **not** grant raw access while another still
masks the column — a narrow explicit exemption cannot disable a broad tag-based net.

## Row-level security

```python
catalog.filter_rows(customers, MatchesAttribute("region", "region"), name="own_region")
catalog.filter_rows(customers, AttributeIn("region", "regions"), exempt=["auditor"])
catalog.filter_rows(customers, lambda p: bt.col("tier") == "public")   # in-process only
```

The predicate takes the **principal**, not a row, and is called once while the plan is
built. `MatchesAttribute(column, attribute)` and `AttributeIn(column, attribute, sep=",")`
are the declarative (picklable) forms; a principal missing the referenced attribute raises
`PlanError` rather than silently admitting or excluding every row.

## Protecting PII

Policy masks lower to the data-plane security expressions. Choose by what you need back:

| Expression        | Reversible?       | Joinable? | Policy factory              |
| ----------------- | ----------------- | --------- | --------------------------- |
| `bt.mask`         | no                | no        | `Redact(show_first, show_last, char)` |
| `bt.hmac_sha256`  | no                | yes       | `Pseudonymize("env:KEY")`   |
| `bt.aes_encrypt`  | yes, with the key | yes       | `Encrypt("env:KEY")`        |
| (typed NULL)      | no                | no        | `Nullify()`                 |

"Joinable" means equal inputs give equal outputs, so the column still groups and
equi-joins. Every Batcher expression must be deterministic (the sequential interpreter is
the correctness oracle), so `aes_encrypt` uses AES-256-GCM-SIV — the AEAD whose security
survives the fixed nonce determinism forces. **The price is that equality is observable:
an encrypted column reveals which rows share a value.** Where that is unacceptable, leave
the column out of the projection entirely.

```python
key = "env:PII_KEY"                                   # a reference, never the secret
users = bt.from_pydict({"email": ["a@x.com", "a@x.com", "b@x.com"]})
out = users.select(p=bt.hmac_sha256(bt.col("email"), key=key)).to_pydict()
print(out["p"][0] == out["p"][1], out["p"][0] == out["p"][2])   # True False — stable, so it joins
```

`env:NAME` / `file:PATH` are resolved by the data plane on the machine that runs the
query, so the secret never enters the plan IR, `explain()`, logs, or the FFI boundary —
and a distributed query reads the key per worker instead of shipping it. An inline literal
key still works locally but emits a `SecurityWarning`. `aes_decrypt` under the wrong key
yields NULL rather than failing the query (one unreadable row must not abort a
billion-row scan), so an all-NULL result is the signal that the key is wrong. Prefer
`hmac_sha256` over a bare digest: a plain hash of a low-entropy value like an email is
recovered by enumeration; an HMAC is not.

**`bt.mask` is a string-redaction expression, not governance.** Calling
`ds.select(e=bt.mask(bt.col("email")))` is ad-hoc redaction: applied wherever you put it,
trivially removed by editing the query, enforcing nothing. Policy masking is a
`ColumnMask`/`TagMask` registered on a `SecurityCatalog` and applied as the plan rewrite
above — that is what makes it unavoidable. `Redact` merely *lowers to* `bt.mask`.

## Auditing

```python
events = []
with bt.security(catalog, analyst, audit=events.append):
    _ = bt.read.parquet(customers).select("id").collect()

ev = events[0]
print(ev.allowed, ev.visible, ev.denied, ev.masked, ev.row_filters)
# True ('id', 'email', 'region', 'amount') ('ssn',) ('email',) ('own_region',)
```

One `GovernanceEvent` per governed table, emitted from the *same traversal* that builds
the governed plan — so the log cannot drift from what was enforced. Every decision is also
logged at INFO regardless of the sink. An event names columns and policies, never values
and never key material, so it is safe to write somewhere the data may not go. It carries
no timestamp (`enforce` is pure); the emitter stamps it.

## Column-level lineage

```python
ds = bt.read.parquet(customers).select(who=bt.col("email"), total=bt.col("amount") * 2)
print(ds.lineage())
# {'who': ['/data/customers.parquet.email'], 'total': ['/data/customers.parquet.amount']}
```

`ds.lineage()` reads the plan and executes nothing. It is what turns a `tag` into an
answer: tag `customers.ssn` once, and lineage names every downstream column carrying it.
It **over-approximates by design** — an opaque `map_batches` is treated as though every
output column derives from every input column, because a false "this might carry PII"
costs a review and a false "this cannot" costs a breach. It tracks *data* flow, not
control flow: filtering on `ssn` does not put `ssn` in the surviving columns' lineage
(matching Unity Catalog and Snowflake). `governance.column_lineage(plan, tables)` is the
underlying analysis.

## Persisting a policy

Lambdas are impossible to store. For a policy your platform keeps externally and rebuilds
per session, use the declarative factories (`Redact`, `Pseudonymize`, `Encrypt`,
`Nullify`, `MatchesAttribute`, `AttributeIn`) — small frozen dataclasses in
`governance/masks.py` and `governance/filters.py` that are callable **and** picklable, so
a catalog built from them round-trips and enforces identically. Batcher persists nothing
itself. Note also that `SecurityCatalog` is **mutable** despite reading fluently: the
declaration methods return `self`, they do not derive a new catalog.

## Correctness stakes — verify, don't assume

Getting PII masking wrong is not recoverable: leaked data cannot be un-leaked, and a
governance bug is silent by construction — the query succeeds and returns rows. Before
trusting a policy, verify all of:

- [ ] **Every read is inside a `bt.security(...)` block.** A read outside one is
      ungoverned and looks completely normal. This is the failure that actually happens.
- [ ] **The rewrite is really there**: `print(ds.explain())` inside the block shows
      `project` over `filter` over `scan`, and the row estimate drops.
- [ ] **Assert on values, not just on the catalog**: collect as the restricted principal
      and check the masked column holds no plaintext and the row set is exactly the
      allowed one. A green `mask_for(...)` proves the policy resolved, not that the query
      applied it.
- [ ] **Test the exempt and the denied principal too** — an exemption that accidentally
      applies to everyone passes every test written for the restricted case.
- [ ] **Check the escape hatches**: a masked column reached through `map_batches`, a join,
      a group-by key, a sort, or a `select` alias. The leaf rewrite is what makes these
      safe; a change that moves masking later breaks all of them at once.
- [ ] **Run lineage** on derived tables: no untagged output column traces back to a
      tagged source column.
- [ ] **Keys are references** (`env:`/`file:`), no `SecurityWarning`, and leaking
      *equality* on an encrypted/pseudonymized column is acceptable here.

Reference tests to extend rather than reinvent: `tests/unit/test_governance*.py` and
`test_lineage.py`; `tests/integration/test_governance_enforcement.py`;
`tests/differential/test_diff_security_functions.py`, `test_diff_gov2_lineage.py`.

## Adding a new mask or row-filter policy class

Both are small frozen dataclasses — no new subsystem, no engine change:

- **Mask** → `governance/masks.py`: `@dataclass(frozen=True, slots=True)` with a
  `__call__(self, column: Expr) -> Expr` built from existing expressions. Keep it
  picklable (plain fields only; a key is a *reference* string, never the secret) and let
  it lower to a data-plane function so the redaction runs in Rust. Add it to `__all__`.
- **Row filter** → `governance/filters.py`: same shape with
  `__call__(self, principal: Principal) -> Expr`. Fail closed on a missing attribute
  (`_require_attr` raises `PlanError`).

`policy.py` holds the records the catalog stores (`Grant`, `ColumnMask`, `TagMask`,
`RowFilter`); touch it only if the *policy kind* is new. Add a unit test that the class
pickles and round-trips, plus a differential test if it introduces a new expression.
`governance` is a layer-3 subsystem: it may import `plan`/`metadata`/`config`/`_internal`
and **must not** import `kyber`, `carbonite`, or `core` (`just lint-layers`).

## See also

- `docs/user-guide/trust/governance.md` (the full narrative), `docs/api/operations/governance.md`,
  `docs/user-guide/trust/data-quality.md` (validate/quarantine rows before a consumer sees them).
- Source: `python/batcher/governance/{catalog,principal,policy,masks,filters,enforce,
  lineage,audit}.py`; `python/batcher/api/security/`.
- Rules: `.claude/rules/architecture.md` (governance is an independent layer-3 subsystem
  that decides and never executes), `.claude/rules/testing.md`.
- Skills: `write-a-batcher-pipeline`; `debug-a-batcher-query` (a governed query returning
  unexpected rows); `run-a-distributed-job`; `run-quality-gate`.
