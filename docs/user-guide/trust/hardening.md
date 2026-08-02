# Hardening a deployment

This page covers the settings to change before running Batcher somewhere that matters, and
and, just as importantly, the boundaries Batcher does not enforce, so you can put a real one
around it.

Read {doc}`/user-guide/trust/governance` first for what row filters and column masks do. This page is about
making them mandatory, and about everything else on the disk and in the process.

## What Batcher enforces, and what it does not

Batcher **authorizes**. It does not **authenticate**.

A `Principal` is asserted by the caller. Any code running inside the engine's process can
construct any principal it likes, including one holding every role, and Batcher will honor
it. That is not an oversight to be fixed by a future release: Batcher is a library imported
into your process, and code already inside a process cannot be kept out of it.

So the trust boundary is **the process**, and the deployment pattern that follows is:

- Run one process per trust domain. Authenticate at the layer that has a network edge,
  meaning your notebook server, API gateway, or job submitter, and pass the identity it
  established into `bt.security(...)`.
- Treat "who is running this query" as answered before Batcher starts, not by Batcher.

Everything below hardens what happens *inside* that boundary.

## Make governance mandatory

By default, row filters and column masks apply only inside a `bt.security(...)` block. A
`Dataset` built outside one is ungoverned, which is the correct default for a library and
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

Under `strict`, a read that no `security()` block covers raises `AccessDeniedError`. So
does a source that cannot be governed at all. An in-memory table or a live stream has no
durable name to write a policy about, so it is refused rather than silently exempted.

```{tip}
Do not skip `advisory`. Switching a live system straight to `strict` fails on the first
pipeline that joins in a dict, and you will find out from a pager rather than a warning.
```

## Require verified identities

By default a `Principal` is whatever the caller says it is. `bt.Principal("root",
roles=["admin"])` holds every admin role, and every policy honours it. For a single-user
session that is fine. For a deployment it means your row filters and column masks can be
stepped around by a constructor call.

Install a verifier at startup, from the layer that owns the network edge, and turn on
`require_verified_principal`:

```python
import dataclasses

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

With that on, entering `bt.security(catalog, principal)` with an asserted principal raises
`AccessDeniedError`. Expired claims are refused whether or not the setting is on, so a
long-running process cannot keep acting on a token that lapsed hours ago.

For a fleet where a submitter authenticates users and hands tokens to workers, use
`HmacTokenVerifier` (standard library only, key resolvable as `env:`/`file:`). For an
existing identity provider, use `JwtVerifier` against its JWKS endpoint.

```{warning}
This is a deployment control, not a security boundary. Code inside the engine's process can
set `issuer` by hand. It makes "we only accept established identities" enforceable for the
code paths you control; it does not make Batcher a trust boundary.
```

## Isolate UDF processes

A `map_batches` UDF that runs on the process pool executes in a child of the engine
process. Children inherit the parent's environment, and that is where credentials live:
`env:` secret references and `BATCHER_SECRET_COMMAND`, which names the helper that fetches
arbitrary secrets on request.

`execution.udf_isolation` controls what a worker child inherits. It defaults to `"env"`,
which rebuilds the child's environment from an allowlist and drops every `BATCHER_*`
variable. Set it to `"strict"` to add resource ceilings:

```python
import dataclasses

from batcher import Config, ExecutionConfig

cfg = Config()
hardened = cfg.replace(
    execution=dataclasses.replace(
        cfg.execution,
        udf_isolation="strict",
        udf_memory_limit_bytes=8 * 1024**3,
        udf_timeout_s=600.0,
    )
)
print(hardened.execution.udf_isolation)
# strict
```

`udf_memory_limit_bytes` becomes an `RLIMIT_AS` on the child, so a runaway allocation
raises `MemoryError` in the guilty worker instead of drawing the kernel's OOM killer onto
whatever else is on the box. `udf_timeout_s` bounds a wedged UDF, which otherwise hangs the
query with no error at all. If a UDF needs a variable the allowlist drops, name it in
`execution.udf_env_allowlist` rather than turning isolation off.

```{warning}
This is defense in depth, not a sandbox, and the difference matters. A UDF is arbitrary
Python and can reach any syscall through `ctypes`. It also covers the *process* path only:
a UDF that runs on a thread executes inside the engine process and can read its
environment, because it is that process.

**Run untrusted UDFs in a container, not behind a config flag.**
```

## Bound how many queries run at once

Batcher admits every arriving query immediately by default, and each one asks the executor for a worker pool sized to every core. That is right for one query and wrong for sixteen: sixteen full-width pools on one machine spend their time context-switching rather than working.

Set `execution.max_concurrent_queries` to bound it:

```python
import dataclasses

from batcher import Config, ExecutionConfig

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

Query five then waits for a slot rather than joining the scrum, and each admitted query requests a proportionally narrower pool, so four concurrent queries divide the machine instead of each claiming all of it. A single query still gets every core.

`admission_queue_depth` caps the waiting line. Past it, a query raises `AdmissionTimeout` immediately instead of joining a queue nobody is draining, which is an outage that presents as slowness. `admission_timeout_s` bounds how long an admitted-but-waiting query blocks.

A `collect()` nested inside a `map_batches` UDF does not consume a second slot. The outer query already holds the machine, and making the inner one queue behind it would deadlock the process against itself.

```{note}
This is a per-process gate. Batcher has no cross-node admission queue, so on a Ray cluster each driver bounds only its own concurrency.
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
{doc}`/user-guide/trust/secrets` for the `env:`, `file:`, and `cmd:` reference schemes. `cmd:` is how you
reach Vault, AWS Secrets Manager, or Google Secret Manager without Batcher linking a cloud
SDK.

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

## Checklist

Before a deployment that matters, complete the following:

1. Decide the trust boundary and run one process per trust domain.
1. Run under `governance.mode="advisory"`, fix every warning, then switch to `"strict"`.
1. Set `execution.udf_isolation="strict"` with a memory limit and a timeout, or run
   untrusted UDFs in a container.
1. Set `execution.max_concurrent_queries` if more than one query runs at a time.
1. Point `memory.spill_dir` at a volume you control, on an encrypted filesystem.
1. Pass every key and credential by reference.
1. Confirm the `MetadataHub` backend's access controls match the data it will hold
   statistics about.

## Requirements and limitations

- Batcher does not authenticate. It consumes an identity from the layer that did.
- Batcher is not multi-tenant. The tenant boundary is the process; a process-global result
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
