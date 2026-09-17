# Secrets and keys

This page describes how to keep an encryption key, a connector password, or an API token
out of your query and out of its plan. Batcher resolves a *reference* to a secret on the
machine that needs it, so the secret itself never travels in the plan IR, a log line, or a
pickled task.

The functions that take a key are covered in {doc}`/user-guide/trust/governance`. This page is about where that key comes from.

## Keys by reference

Pass a key reference rather than the raw key:

```python
# docs: skip
enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), "env:AES_KEY"))  # from the environment
enc = ds.select(c=bt.aes_encrypt(bt.col("ssn"), "file:/run/secrets/aes"))  # from a mounted secret
```

`env:NAME` reads an environment variable and `file:PATH` reads a mounted secret file. Only
the reference travels in the plan IR. Plan logs, the profile, `explain()`, and the FFI
boundary never see the secret, and the data plane resolves it on the machine that runs the
query, so a distributed query reads the key on each worker rather than shipping it over the
wire.

The reference and the key take different routes, and only one of them leaves the machine it started on:

![Three columns: the driver, what travels, and each worker. On the driver, ds.select(c=bt.aes_encrypt(bt.col('ssn'), 'env:AES_KEY')) lowers to the plan IR, which is shipped to every worker. The plan carries only the reference env:AES_KEY, never the key, so plan logs, the profile, explain() and the FFI boundary never see the key. An inline key instead of a reference puts the key in the plan and raises a SecurityWarning, or a PlanError at plan-build time when BATCHER_REQUIRE_KEY_REFS=1 is set. When the plan arrives on a worker, bc-secrets resolves the reference in the data plane by its scheme: env: reads an environment variable, file: reads a mounted file, and cmd: takes the stdout of the operator's BATCHER_SECRET_COMMAND run with NAME as its argument, failing if that variable is unset. The secret is cached per process for BATCHER_SECRET_TTL_SECONDS, 300 seconds by default, because key references resolve per batch, and the kernel uses the key on the machine that runs it. A missing reference fails naming the reference, never the key. Connector passwords and storage_options take the same references, resolved once per connection and not cached.](/_static/diagrams/secret_reference_flow.svg)

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

bt.read.clickhouse(
    query="SELECT ...", host="ch.internal", database="events", password="env:CH_PASSWORD"
)

bt.read.table("connectorx", query="SELECT ...", conn_uri="file:/run/secrets/pg_uri")

bt.read.mongo(uri="env:MONGO_URI", database="app", collection="events")

bt.read.parquet(
    "oss://bucket/events/*.parquet",
    storage_options={"key": "env:OSS_KEY", "secret": "cmd:prod/oss-secret"},
)
```

All three schemes work for connector options, including `cmd:`, and so do the `storage_options` an object store takes. A connector resolves once when it opens its connection rather than per batch,
so there is no cache in this path and a rotated secret is picked up by the next connection.

The reference is resolved on the machine that *opens the connection*, not on the driver
that builds the plan. The source object and the pickled split that reaches a Ray worker
carry only the reference, so the secret never crosses the wire, never sits in driver
memory, and cannot surface in a traceback or a log line that renders a split.

A literal password still works unchanged. This is additive, not a migration.

## Reaching Vault, KMS, or Secret Manager

Two schemes cover an external key store from anywhere in the engine, including the
expression layer, and neither links a cloud SDK into it.

The first is a file the platform delivers. Vault Agent, the External Secrets Operator, and
the Kubernetes secrets-store CSI driver all materialize a secret as a file, so
`file:/run/secrets/aes-key` *is* the integration. Rotation, authentication, and audit stay
with the platform that owns them.

The second is `cmd:NAME`, which goes through a helper program. Batcher runs the
operator-configured `BATCHER_SECRET_COMMAND` with `NAME` as its argument and takes stdout as
the secret:

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

### Naming a key store directly

Connector credentials and storage options accept five further schemes that read a key
store without a helper program. These are the shim most teams write around `cmd:`, written
once:

| Scheme | Example | Reads |
|---|---|---|
| `vault:` | `vault:secret/data/warehouse#password` | HashiCorp Vault KV, v1 or v2 |
| `aws-sm:` | `aws-sm:prod/warehouse#password` | AWS Secrets Manager |
| `aws-ssm:` | `aws-ssm:/prod/warehouse/password` | AWS SSM Parameter Store |
| `gcp-sm:` | `gcp-sm:projects/p/secrets/db-password` | Google Cloud Secret Manager |
| `azure-kv:` | `azure-kv:https://v.vault.azure.net/secrets/db` | Azure Key Vault |

```python
# docs: skip
ds = bt.read.sql(
    "SELECT * FROM orders",
    uri="postgresql://analytics@warehouse.internal/sales",
    password="aws-sm:prod/warehouse#password",
)
```

The `#key` suffix selects a field when the stored secret is a JSON object, which is what
the Secrets Manager console writes for anything with more than one field, and which key of
a Vault path is meant.

Each of these resolves on the worker that opens the connection, against that machine's own
identity. On a distributed query only the reference travels in the split, and each worker
authenticates as itself through an instance profile, an IRSA role, a Workload Identity
binding, or a Vault Kubernetes login. A node without an identity fails closed. It does not
inherit the driver's. This is the same property the `env:`/`file:`/`cmd:` schemes have, and
the reason all of them are references rather than values.

Vault needs no extra package: its read is one authenticated GET. The three cloud stores
need their vendor SDK, so install `batcher-engine[aws-secrets]`,
`batcher-engine[gcp-secrets]`, or `batcher-engine[azure-secrets]`.

Vault takes its address from `VAULT_ADDR` and its token from `VAULT_TOKEN`, the same
variables the Vault CLI reads. With no token set, and `VAULT_K8S_ROLE` set instead,
Batcher exchanges the pod's projected service account token at `auth/kubernetes/login`,
which is the route that needs no long-lived credential on any worker.

The expression layer's key references (`aes_encrypt` and friends) resolve in the data
plane, which deliberately links no cloud SDK, so they keep to `env:`, `file:`, and `cmd:`.
Point `BATCHER_SECRET_COMMAND` at a helper to reach a key store from there.

### Caching

Key references resolve on a per-batch path, so they are cached for
`BATCHER_SECRET_TTL_SECONDS` (default 300, `0` disables). Without a cache, a `cmd:` reference
would fork a process for every Arrow batch. The TTL bounds how long a rotated secret stays
stale. Connector credentials and storage options are not cached, because they resolve once
per connection.

## Enforcing references in a regulated deployment

The warning is a weak control on its own. `SecurityWarning` is a `UserWarning`, so Python
prints it once per call site and a process that filtered warnings never sees it. Meanwhile
an inline key still travels verbatim in the serialized IR, into `explain(format="json")`
and the plan fingerprint, and out to every worker the plan is shipped to.

Set `BATCHER_REQUIRE_KEY_REFS=1` to refuse inline keys outright. {py:func}`aes_encrypt <batcher.aes_encrypt>`,
{py:func}`aes_decrypt <batcher.aes_decrypt>`, and {py:func}`hmac_sha256 <batcher.hmac_sha256>` then raise {py:exc}`PlanError <batcher.PlanError>` at plan-build time unless the key
is an `env:`, `file:`, or `cmd:` reference. Set it in the pod spec or node environment for the whole
deployment, and leave it unset in notebooks and tests, where an inline key is legitimate.

```bash
export BATCHER_REQUIRE_KEY_REFS=1
```

:::{note}
Prefer `file:` over `env:` where a user-supplied UDF may run. A UDF on the process pool runs in a child whose environment is rebuilt from an allowlist under the default `execution.udf_isolation="env"`, but a UDF that runs on a thread executes inside the engine process and can read its environment. A `file:` reference with restrictive permissions is not readable the same way. See {doc}`/user-guide/trust/hardening`.
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
- {doc}`/user-guide/trust/hardening`: the rest of a production deployment, including UDF isolation, which keeps `env:` material away from user code.
- {doc}`/user-guide/moving-data/reading-data`: the connectors whose credentials take the same references.
- {doc}`/configuration/options`: `memory.spill_dir` and the rest of the configuration.
- {doc}`/cookbook/governance/pii_transforms`: masking, hashing, and encrypting a column, as a script.
