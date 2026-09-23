# Hardening a deployment

This page covers the settings to change before running Batcher somewhere that matters. It
also covers the boundaries Batcher does not enforce, so you can put a real one around it.

Read {doc}`/user-guide/trust/governance` first for what row filters and column masks do. This page is about
making them mandatory, and about everything else on the disk and in the process.

## What Batcher enforces, and what it does not

Batcher **authorizes**. It does not **authenticate**.

A {py:class}`Principal <batcher.Principal>` is asserted by the caller. Any code running inside the engine's process can
construct any principal it likes, including one holding every role, and Batcher will honor
it. That is not an oversight to be fixed by a future release: Batcher is a library imported
into your process, and code already inside a process cannot be kept out of it.

So the trust boundary is **the process**, and the deployment pattern that follows is:

- Run one process per trust domain. Authenticate at the layer that has a network edge,
  meaning your notebook server, API gateway, or job submitter, and pass the identity it
  established into {py:func}`bt.security(...) <batcher.security>`.
- Treat "who is running this query" as answered before Batcher starts, not by Batcher.

Everything below hardens what happens *inside* that boundary. Each layer is only as strong as the one around it:

![Three nested layers. The outer layer is what your platform provides and Batcher does not: authentication at your network edge, tenant isolation by running one process per trust domain, sandboxing untrusted UDFs in containers, and encryption at rest on an encrypted volume. Inside it, the process is the trust boundary. Within the process Batcher enforces eight things: governance required with governance.mode='strict', a durable audit trail with governance.audit_path, verified identities with require_verified_principal, key references only with BATCHER_REQUIRE_KEY_REFS=1, UDF process ceilings with udf_isolation='strict', admission control with max_concurrent_queries, owner-only artifacts with 0700 directories and 0600 files always, and governed statistics that keep no min or max for masked columns. Even inside the process it cannot guarantee three things: code in the process can construct any Principal, a UDF on a thread reads the engine's environment, and admission bounds one process rather than the cluster. So authenticate at the edge, pass the identity into bt.security(), and run untrusted UDFs in a container.](/_static/diagrams/hardening_boundary.svg)

## Make governance mandatory

By default, row filters and column masks apply only inside a {py:obj}`bt.security(...) <batcher.security>` block. A
{py:class}`Dataset <batcher.Dataset>` built outside one is ungoverned, which is the correct default for a library and
the wrong one for a deployment: forgetting the `with` block becomes the difference between
a masked column and a plain one, and nothing says so.

`governance.mode` closes that. Move through it in two steps:

```python
from batcher import Config, GovernanceConfig

# Step 1: find the ungoverned reads without breaking anything.
advisory = Config().replace(governance=GovernanceConfig(mode="advisory"))
print(advisory.governance.mode)
# advisory
```

Run your real workload under `advisory`. Every read that a strict deployment would refuse
raises a `SecurityWarning` naming the source. Fix them, then switch:

```python
from batcher import Config, GovernanceConfig

strict = Config().replace(governance=GovernanceConfig(mode="strict"))
print(strict.governance.mode)
# strict
```

Under `strict`, a read that no {py:func}`security() <batcher.security>` block covers raises {py:exc}`AccessDeniedError <batcher.AccessDeniedError>`. So
does a source that cannot be governed at all. An in-memory table or a live stream has no
durable name to write a policy about, so it is refused rather than silently exempted.

```{tip}
Don't skip `advisory`. Switching a live system straight to `strict` fails on the first
pipeline that joins in a dict, and you will find out from a pager rather than a warning.
```

## Keep a durable audit trail

Every governed read and write emits a `GovernanceEvent`, and by default it goes to the `batcher.governance` logger and to any `audit=` callback you pass to {py:obj}`bt.security(...) <batcher.security>`. Neither is an audit trail a reviewer can rely on, because a caller can simply not pass the callback.

Set `governance.audit_path` to make the record unconditional:

```python
import os
import tempfile

from batcher import Config, GovernanceConfig

audited = Config().replace(
    governance=GovernanceConfig(mode="strict", audit_path=os.path.join(tempfile.mkdtemp(), "audit.jsonl"))
)
print(audited.governance.audit_path.endswith("audit.jsonl"))
# True
```

Batcher appends one JSON line per decision, naming the principal, its roles, the table, the privilege, and the visible, denied, and masked columns. The file is created owner-only and reopened for each record, so `logrotate` can rotate it underneath a running engine. A record that can't be written fails the read or write it describes, rather than letting the access go unrecorded.

## Require verified identities

By default a `Principal` is whatever the caller says it is. `bt.Principal("root",
roles=["admin"])` holds every admin role, and every policy honours it. For a single-user
session that is fine. For a deployment it means your row filters and column masks can be
stepped around by a constructor call.

Install a verifier at startup, from the layer that owns the network edge, and turn on
`require_verified_principal`:

```python
import batcher as bt
from batcher import Config, GovernanceConfig
from batcher.governance.authn import ProcessIdentityVerifier

# One process per trust domain: the OS already answered "who is this".
bt.set_verifier(ProcessIdentityVerifier(roles={"analyst"}))
print(bt.current_verifier() is not None)
# True

principal = bt.authenticate()
print(principal.name == __import__("getpass").getuser(), principal.verified)
# True True

strict = Config().replace(governance=GovernanceConfig(require_verified_principal=True))
print(strict.governance.require_verified_principal)
# True

bt.set_verifier(None)
```

With that on, entering {py:obj}`bt.security(catalog, principal) <batcher.security>` with an asserted principal raises
`AccessDeniedError`. Expired claims are refused whether or not the setting is on, so a
long-running process cannot keep acting on a token that lapsed hours ago.

For a fleet where a submitter authenticates users and hands tokens to workers, use
`HmacTokenVerifier` (standard library only, key resolvable as `env:`/`file:`). For an
existing identity provider, use `JwtVerifier`.

Name the issuer and let Batcher find its keys, rather than looking up a JWKS URL and
pasting it into another config file:

```python
# docs: skip
import batcher as bt
from batcher.governance.authn import JwtVerifier

bt.set_verifier(
    JwtVerifier.from_issuer("https://login.microsoftonline.com/<tenant>/v2.0", audience="batcher")
)
```

`from_issuer` reads the provider's OIDC metadata at
`<issuer>/.well-known/openid-configuration` and binds to the `jwks_uri` it publishes. A
provider that does not publish metadata still works: pass `jwks_url` to the constructor
directly.

Both the metadata and the key set are cached per process, and each worker on a distributed
query fetches for itself against its own network path to the provider. That matters for
more than latency. A key set shipped from the driver would be the driver vouching for the
issuer, which is the trust hop the verifier exists to remove, and it would make the
identity provider a hard dependency of every query rather than of every five minutes.

Signature algorithms default to asymmetric only. That default is load-bearing: allowing
`HS256` alongside `RS256` is the algorithm-confusion attack, where an attacker signs a
token with the public key used as an HMAC secret.

**Set the issuer and the audience.** Both default to empty and both are then skipped, and
what that costs is a property of how the large identity providers are deployed: they publish
one key set across many tenants and many applications. A valid signature proves which key
set signed the token, never who it was minted for. So with `iss` unchecked a token from
another tenant of the same provider verifies, and with `aud` unchecked a token minted for
another application of the same tenant verifies. Both are real tokens, correctly signed,
issued to somebody else and replayed at you.

`from_issuer` sets the issuer for you. Nothing can guess the audience, so pass it. Leaving
either unset stays legal, because a deployment mid-migration may not know its audience yet,
and raises a `SecurityWarning` naming the check that was skipped and what verifies without
it.

```{warning}
This is a deployment control, not a security boundary. Code inside the engine's process can
set `issuer` by hand. It makes "we only accept established identities" enforceable for the
code paths you control. It does not make Batcher a trust boundary.
```

## Isolate UDF processes

A `map_batches` UDF that runs on the process pool executes in a child of the engine
process. Children inherit the parent's environment, and that is where credentials live:
`env:` secret references and `BATCHER_SECRET_COMMAND`, which names the helper that fetches
arbitrary secrets on request.

`execution.udf_isolation` controls what a worker child inherits. It defaults to `"env"`,
which rebuilds the child's environment from an allowlist and drops every `BATCHER_*`
variable. `"none"` lets the child inherit everything, for an embedder whose UDFs are as trusted as its own code. Set it to `"strict"` to add resource ceilings:

```python
import dataclasses

from batcher import Config

cfg = Config()
hardened = cfg.replace(
    execution=dataclasses.replace(
        cfg.execution,
        udf_isolation="strict",
        udf_memory_limit_bytes=8 * 1024**3,
        udf_cpu_limit_seconds=300,
        udf_timeout_s=600.0,
    )
)
print(hardened.execution.udf_isolation)
# strict
```

`udf_memory_limit_bytes` becomes an `RLIMIT_AS` on the child, so a runaway allocation
raises `MemoryError` in the guilty worker instead of drawing the kernel's OOM killer onto
whatever else is on the box. `udf_cpu_limit_seconds` becomes an `RLIMIT_CPU`, which bounds
a UDF that is spinning rather than allocating, and does so in the kernel so the driver does
not have to be watching. `udf_timeout_s` bounds a wedged UDF by wall clock, which otherwise
hangs the query with no error at all. The two ceilings answer different failures, so a
strict deployment usually wants both. If a UDF needs a variable the allowlist drops, name
it in `execution.udf_env_allowlist` rather than turning isolation off.

```{warning}
This is defense in depth, not a sandbox, and the difference matters. A UDF is arbitrary
Python and can reach any syscall through `ctypes`. It also covers the *process* path only:
a UDF that runs on a thread executes inside the engine process and can read its
environment, because it is that process.

**Run untrusted UDFs in a container, not behind a config flag.**
```

## Bound how many queries run at once

Batcher admits every arriving query immediately by default, and each one asks the executor
for a worker pool sized to every core. That is right for one query and wrong for sixteen.
Sixteen full-width pools on one machine spend their time context-switching rather than
working.

Set `execution.max_concurrent_queries` to bound it:

```python
import dataclasses

from batcher import Config

cfg = Config()
bounded = cfg.replace(
    execution=dataclasses.replace(
        cfg.execution,
        max_concurrent_queries=4,
        admission_queue_depth=200,
        admission_timeout_s=30.0,
    )
)
print(bounded.execution.max_concurrent_queries)
# 4
```

Query five then waits for a slot rather than joining the scrum, and each admitted query
requests a proportionally narrower pool, so four concurrent queries divide the machine
instead of each claiming all of it. A single query still gets every core.

`admission_queue_depth` caps the waiting line. Past it, a query raises `AdmissionTimeout`
immediately instead of joining a queue nobody is draining, which is an outage that presents
as slowness. `admission_timeout_s` bounds how long an admitted-but-waiting query blocks.

A `collect()` nested inside a `map_batches` UDF does not consume a second slot. The outer
query already holds the machine, and making the inner one queue behind it would deadlock the
process against itself.

```{note}
This is a per-process gate. Batcher has no cross-node admission queue, so on a Ray cluster
each driver bounds only its own concurrency.
```

## Artifacts on disk

Batcher writes several things to disk, and none of them are metadata:

| Artifact | What it contains |
|---|---|
| Spill files | The query's actual rows |
| Shuffle scratch | The query's actual rows, often on a shared cluster mount |
| UDF input shards | The query's actual batches, in `/dev/shm` |
| Event-log documents | The whole plan, including literal predicate constants |
| Learned-stats database | Persisted column statistics, including `min`/`max` |

All of them are created owner-only (`0700` directories, `0600` files). You do not need to
configure that, but you should know it is the whole of the at-rest protection: Batcher does
not encrypt these files. Pair it with full-disk or volume encryption, which is what
actually protects the bytes if the disk leaves the building.

Point `memory.spill_dir` at a volume you control rather than a shared `/tmp`.

## Keep secrets out of plans

Pass keys and credentials by reference, never inline. An inline key is embedded in the
query plan, and therefore in any plan log, profile, or `explain()` output. See
{doc}`/user-guide/trust/secrets` for the `env:`, `file:`, and `cmd:` reference schemes, and for the `vault:`, `aws-sm:`, `aws-ssm:`, `gcp-sm:`, and `azure-kv:` schemes that connector credentials accept. Set `BATCHER_REQUIRE_KEY_REFS=1` to refuse an inline key outright.

## Statistics and governed columns

Batcher persists per-column statistics into the `MetadataHub` so later queries plan better.
When the hub is shared across a fleet (the Redis or object-storage backends), so is
everything in it.

Inside a `security()` block, columns that are masked or invisible to the running principal
keep only their cardinalities: row counts, null counts, and distinct estimates. Value-derived
statistics are dropped, because a bloom filter over a governed column answers membership
questions about its values, and `min`/`max` are two of those values outright.

This happens automatically. It is worth knowing because it is a reason to *use* the
`security()` block even for a pipeline that only writes.

## Know what a partial mask does to a short value

`Redact` reveals the first or last few characters of a value, which is the "card ending 1234"
pattern. A value no longer than what the policy reveals is masked **completely** rather than
returned as it is.

That case is not rare. A masking policy is usually written about names, postcodes, national
identifiers and country codes, and many of the values in such a column are shorter than the
four characters a card policy reveals:

```python
import batcher as bt
from batcher.governance import Redact

people = bt.from_pydict({"name": ["Anastasia", "Bo", "Li"], "postcode": ["SW1A 2AA", "EC1", "N1"]})
masked = people.select(
    name=Redact(show_first=1)(bt.col("name")),
    postcode=Redact(show_last=3)(bt.col("postcode")),
)
print(masked.to_pydict())
```

```text
{'name': ['AXXXXXXXX', 'BX', 'LX'], 'postcode': ['XXXXX2AA', 'XXX', 'XX']}
```

`EC1` and `N1` are no longer than the three characters the postcode policy reveals, so they
come back fully masked. `Bo` and `Li` are longer than the single character the name policy
reveals, so they keep their initial.

Two things follow when you write a policy:

- Choose `show_first` and `show_last` against the *shortest* values you expect, not the
  longest. Revealing four characters of a column whose median value is five characters is a
  policy that mostly discloses.
- Masking is length-preserving, so the output still tells a reader how long the value was.
  Where the length itself is sensitive, reach for {py:class}`Pseudonymize
  <batcher.governance.Pseudonymize>` or {py:class}`Nullify <batcher.governance.Nullify>`
  instead.

## Checklist

Before a deployment that matters, complete the following:

1. Decide the trust boundary and run one process per trust domain.
1. Run under `governance.mode="advisory"`, fix every warning, then switch to `"strict"`.
1. Set `execution.udf_isolation="strict"` with a memory limit and a timeout, or run
   untrusted UDFs in a container.
1. Set `execution.max_concurrent_queries` if more than one query runs at a time.
1. Point `memory.spill_dir` at a volume you control, on an encrypted filesystem.
1. Set `governance.audit_path` so every decision lands in a file you keep.
1. Pass every key and credential by reference, and set `BATCHER_REQUIRE_KEY_REFS=1`.
1. Confirm the `MetadataHub` backend's access controls match the data it will hold
   statistics about.
1. Size every partial mask against the shortest values in its column, not the longest.

## Requirements and limitations

- Batcher does not authenticate. It consumes an identity from the layer that did.
- Batcher is not multi-tenant. The tenant boundary is the process, and a process-global result
  cache, plan cache, and UDF pool are shared by everything in it.
- Batcher does not encrypt artifacts at rest. It makes them owner-only and expects
  filesystem-level encryption underneath.
- UDF isolation covers the process path, not the thread path, and is not a sandbox.
- Admission is per-process. There is no cross-node queue, so each driver bounds only itself.

## See also

- {doc}`/user-guide/trust/governance`: writing the row filters and column masks this page makes mandatory.
- {doc}`/user-guide/trust/secrets`: reference schemes for keys and credentials.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a query, and where.
- {doc}`/user-guide/moving-data/cloud-storage`: how credentials reach an object store in the first place.
- {doc}`/cookbook/governance/index`: the masking and lineage recipes behind these settings.
