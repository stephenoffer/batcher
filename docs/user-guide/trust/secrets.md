# Secrets and keys

This page describes how to keep an encryption key, a connector password, or an API token
out of your query and out of its plan. Batcher resolves a *reference* to a secret on the
machine that needs it, so the secret itself never travels in the plan IR, a log line, or a
pickled task.

The functions that take a key are covered in {doc}`/user-guide/trust/governance`. This page is about where
that key comes from.

## Keys by reference

Pass a key reference rather than the raw key:

```python
# docs: skip
enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), "env:AES_KEY"))            # from the environment
enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), "file:/run/secrets/aes"))  # from a mounted secret
```

`env:NAME` reads an environment variable and `file:PATH` reads a mounted secret file. Only
the reference travels in the plan IR, so plan logs, the profile, `explain()`, and the FFI
boundary never see the secret. The data plane resolves it on the machine that runs the
query, so a distributed query reads the key on each worker instead of shipping it over the
wire.

A `file:` reference is an ordinary path, so the round trip runs anywhere:

```python
import pathlib

import batcher as bt

pathlib.Path("aes.key").write_text("00" * 32)  # 32 bytes as hex; your platform mounts this

people = bt.from_pydict({"ssn": ["123-45-6789", "987-65-4321"]})
encrypted = people.select(c=bt.aes_encrypt(bt.col("ssn"), "file:aes.key"))
print(encrypted.select(s=bt.aes_decrypt(bt.col("c"), "file:aes.key")).to_pydict())
# {'s': ['123-45-6789', '987-65-4321']}
```

No `SecurityWarning` is raised here, because no key entered the plan.

An inline literal key still works for local development but emits a `SecurityWarning`,
because it embeds the secret in the query and its serialized plan. A missing reference (an
unset `env:` variable, an absent `file:` path) fails loudly, naming the *reference*, never
the key.

## Connection credentials

The same indirection works for every connector password, token, API key, and connection
URI. That is the larger secret surface in most deployments.

```python
# docs: skip
import batcher as bt

bt.read.clickhouse(query="SELECT ...", host="ch.internal", database="events",
                   password="env:CH_PASSWORD")

bt.read.table("connectorx", query="SELECT ...", conn_uri="file:/run/secrets/pg_uri")

bt.read.mongo(uri="env:MONGO_URI", database="app", collection="events")
```

The reference is resolved on the machine that *opens the connection*, not on the driver
that builds the plan. The source object and the pickled split that reaches a Ray worker
carry only the reference, so the secret never crosses the wire, never sits in driver
memory, and cannot surface in a traceback or a log line that renders a split.

A literal password still works unchanged. This is additive, not a migration.

## Reaching Vault, KMS, or Secret Manager

Two schemes cover an external key store, and neither links a cloud SDK into the engine.

**A file, via the platform's own secret delivery.** Vault Agent, the External Secrets
Operator, and the Kubernetes secrets-store CSI driver all materialize a secret as a file,
so `file:/run/secrets/aes-key` *is* the integration. Rotation, authentication, and audit
stay with the platform that owns them.

**`cmd:NAME`, via a helper program.** Batcher runs the operator-configured
`BATCHER_SECRET_COMMAND` with `NAME` as its argument and takes stdout as the secret:

```bash
export BATCHER_SECRET_COMMAND=/usr/local/bin/fetch-secret   # your wrapper around
                                                            # vault / aws / gcloud / az
```

```python
# docs: skip
ds.select(c=bt.aes_encrypt(bt.col("ssn"), "cmd:prod/aes-key"))
```

`cmd:` is inert unless the operator sets `BATCHER_SECRET_COMMAND`, and the reference
supplies only the *argument*, never the program. That asymmetry is the security property.
A plan is data and may arrive from somewhere less trusted than the cluster, so letting it
name a program to execute would turn a secret reference into arbitrary code execution. The
argument is passed as an argument, never through a shell, so metacharacters in a reference
are inert.

Resolutions are cached for `BATCHER_SECRET_TTL_SECONDS` (default 300, `0` disables),
because references resolve on a per-batch path. Without a cache, a `cmd:` reference would
fork a process for every Arrow batch. The TTL bounds how long a rotated secret stays stale.

## Enforcing references in a regulated deployment

The warning is a weak control on its own. `SecurityWarning` is a `UserWarning`, so Python
prints it once per call site and a process that filtered warnings never sees it. Meanwhile
an inline key still travels verbatim in the serialized IR, into `explain(format="json")`
and the plan fingerprint, and out to every worker the plan is shipped to.

Set `BATCHER_REQUIRE_KEY_REFS=1` to refuse inline keys outright. `aes_encrypt`,
`aes_decrypt`, and `hmac_sha256` then raise `PlanError` at plan-build time unless the key
is an `env:` or `file:` reference. Set it in the pod spec or node environment for the whole
deployment, and leave it unset in notebooks and tests, where an inline key is legitimate.

```bash
export BATCHER_REQUIRE_KEY_REFS=1
```

:::{note}
Prefer `file:` over `env:` where a user-supplied UDF may run. A UDF executes in a worker
process that inherits the environment, so it can read an `env:`-referenced key. A `file:`
reference with restrictive permissions is not readable the same way.
:::

## Data at rest on the node

A query that spills writes its actual rows to the local scratch directory. A large
aggregate, join, sort, or window can all do this. Batcher creates that directory `0o700`,
so another local user on a shared node cannot read a spilled join off disk.

That is access control, not encryption: the bytes on disk are plaintext Arrow IPC. If your
threat model includes the disk itself (a seized volume, a snapshot, a multi-tenant host you
do not control), use an encrypted filesystem or an encrypted instance volume for
`memory.spill_dir`. Column-level `aes_encrypt` protects a column end to end, including
through a spill, but costs a decrypt wherever the value is used.

## Requirements and limitations

- A reference is resolved where it is used, so every worker that runs the query needs
  access to the same environment variable, file, or helper command.
- `cmd:` requires the operator to set `BATCHER_SECRET_COMMAND`. Without it, a `cmd:`
  reference fails rather than falling back.
- Nothing here encrypts the spill directory. That is a filesystem or volume decision.

## See also

- {doc}`/user-guide/trust/governance`: the masking, row filters, and audit trail these keys feed.
- {doc}`/user-guide/moving-data/reading-data`: the connectors whose credentials take the same references.
- {doc}`/configuration/options`: `memory.spill_dir` and the rest of the configuration.
- {doc}`/cookbook/governance/pii_transforms`: masking, hashing, and encrypting a column, as a script.
