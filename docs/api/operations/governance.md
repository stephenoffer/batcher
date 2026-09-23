# Governance

Row filters, column masks, and lineage. Governance is a plan rewrite rather than a runtime
check. {py:obj}`enforce <batcher.governance.enforce>` rewrites the `LogicalPlan` before it
executes, so a column a principal may not see never enters the plan and is never read.
There is no filtering pass after the fact, no privileged bypass to forget, and no execution
path that can skip enforcement. That is also why a policy costs a pushed-down filter rather
than a per-row callback.

```python
from batcher.governance import Principal, SecurityCatalog, Grant, Redact, enforce
```

The {doc}`governance guide </user-guide/trust/governance>` is the worked introduction. This
page is the symbol reference.

## Identity

A query runs *as* a {py:obj}`Principal <batcher.Principal>`: a name, the roles
it holds, and its attributes. Attributes are what row filters compare against, so one
policy (`region = principal.attrs["region"]`) serves every user.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   Principal
   GovernanceEvent
```

## Establishing an identity

A `Principal` you construct is *asserted*: a name someone typed. That is right for a
single-user session and worthless as a control, because `bt.Principal("root",
roles=["admin"])` holds every admin role.

A `Principal` from {py:func}`bt.authenticate <batcher.authenticate>` is *verified*: its
claims came out of a credential this process checked. Install a verifier once at startup,
from whichever layer owns the network edge, then set
`governance.require_verified_principal` to refuse asserted identities.

```python
import os

import batcher as bt
from batcher.governance.authn import HmacTokenVerifier

# In production the operator sets this; the reference keeps the key out of the plan.
os.environ["BATCHER_TOKEN_KEY"] = "a-signing-key"

verifier = HmacTokenVerifier(key="env:BATCHER_TOKEN_KEY", issuer="gateway")
bt.set_verifier(verifier)

# The gateway mints a token after authenticating the user; the engine checks it.
token = verifier.mint("ana", roles=["analyst"], ttl_seconds=900)
principal = bt.authenticate(token)
print(principal.name, sorted(principal.roles), principal.verified)
# ana ['analyst'] True

bt.set_verifier(None)
```

Batcher ships three verifiers. {py:obj}`ProcessIdentityVerifier
<batcher.governance.authn.ProcessIdentityVerifier>` reports the OS user, which is the
honest answer when each trust domain runs its own process.
{py:obj}`HmacTokenVerifier <batcher.governance.authn.HmacTokenVerifier>` checks a compact
signed token against a shared key, using only the standard library. {py:obj}`JwtVerifier
<batcher.governance.authn.JwtVerifier>` validates an OIDC ID token against the provider's
JWKS, and needs the optional `pyjwt` dependency. For anything else, implement
{py:obj}`CredentialVerifier <batcher.governance.authn.CredentialVerifier>`, which is the
protocol all three satisfy.

```{eval-rst}
.. currentmodule:: batcher.governance.authn

.. autosummary::
   :toctree: generated
   :nosignatures:

   ProcessIdentityVerifier
   HmacTokenVerifier
   JwtVerifier
   CredentialVerifier
```

```{warning}
Verification is a deployment control rather than a security boundary. Code running inside the
engine's process can construct a `Principal` with any `issuer` it likes, and no in-process
mechanism can stop it. It buys one thing: a query whose identity nobody established is
refused instead of silently trusted. The boundary is still the process, so run one per
trust domain. See {doc}`/user-guide/trust/hardening`.
```

## The catalog

{py:obj}`SecurityCatalog <batcher.SecurityCatalog>` holds the policy: which
roles hold which privilege on which columns, which columns are masked, and which rows
each principal may see.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   SecurityCatalog
   Grant
```

### Privileges

A grant carries one of `PRIVILEGES`, the four SQL privileges spelled the way Snowflake
and Unity Catalog spell them. `SELECT` governs reads and is the default. `INSERT`,
`UPDATE`, and `DELETE` govern writes, and which of them a write needs follows from what
it does to the rows already in the destination. The {doc}`governance guide
</user-guide/trust/governance>` has the table.

A table nobody has granted anything on is open. Once any grant names it, every privilege
on it is deny-by-default, so granting one privilege never confers another: a role given
`INSERT` can add rows and cannot drop them.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autodata:: PRIVILEGES
   :annotation:
```

### Denials

A `Denial` refuses a privilege regardless of what any grant says, the same precedence
`DENY` has in SQL Server and Unity Catalog. It exists because grants *union* across a
principal's roles, which leaves two things unsayable. One is "every column except `salary`",
whose complement is wrong as soon as a column is added. The other is a hard block on a role
that another role's grant would otherwise union around.

Declare one with `SecurityCatalog.deny`, and withdraw a grant with
`SecurityCatalog.revoke`. The two are not interchangeable: `revoke` removes a rule, so a
later grant restores access, while `deny` adds one that a later grant does not override.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   Denial
```

## Row filters

A row filter restricts a table to the rows a principal is allowed to see. The predicate
is evaluated against the *principal*, not the row, so it lowers into the plan as an
ordinary pushed-down filter and costs nothing extra.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   RowFilter
   MatchesAttribute
   AttributeIn
```

## Column masks

A mask changes how a column *reads* rather than whether it reads at all. An analyst sees
`XXXX1234`, while the fraud team sees the number. Bind a mask to one column with
{py:obj}`ColumnMask <batcher.governance.ColumnMask>`, or to a *tag* with
{py:obj}`TagMask <batcher.governance.TagMask>` so it applies wherever that tag
appears, including in tables added later.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   ColumnMask
   TagMask
```

### Mask functions

The masking primitives themselves. {py:obj}`Pseudonymize <batcher.governance.Pseudonymize>`
is deterministic, so masked values still join and group correctly. {py:obj}`Encrypt <batcher.governance.Encrypt>` is reversible with the key, and {py:obj}`Nullify <batcher.governance.Nullify>` isn't.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   Redact
   Nullify
   Pseudonymize
   Encrypt
```

## Enforcement and lineage

{py:obj}`enforce <batcher.governance.enforce>` returns the rewritten plan together with a
{py:obj}`GovernanceEvent <batcher.GovernanceEvent>` for every rule it applied.
Both come out of one traversal, so the audit record is by construction the enforcement.
{py:obj}`column_lineage <batcher.governance.column_lineage>` traces each output column back
to the source columns it derives from. That is how a tag on a source column keeps masking a
value three transformations downstream, after it has been renamed and cast and aggregated.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   enforce
   column_lineage

.. autodata:: Origin
```

## Data residency

A residency rule answers a different question from a grant. A grant says who may read a
dataset. A residency rule says *where it may be computed*. That second half of a
sovereignty obligation is the one a scheduler can break silently, by placing a stage in
whichever region has spare accelerator capacity.

{py:obj}`ResidencyCatalog <batcher.governance.ResidencyCatalog>` holds the rules and resolves
a placement to a {py:obj}`ResidencyVerdict <batcher.governance.ResidencyVerdict>`. Its `mode`
is one of `RESIDENCY_MODES`: `off` checks nothing, `advisory` reports a refusal a caller logs
and proceeds past, and `strict` raises. An unregistered dataset is unrestricted, because
residency is an obligation you state rather than one Batcher infers from a bucket name.

```python
from batcher.governance import DataResidency, ResidencyCatalog

catalog = ResidencyCatalog(mode="strict")
catalog.register(DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR Art. 44"))

verdict = catalog.check("s3://eu-customers/orders", "us-east-1")
print(verdict.allowed)
# False
print(verdict.message())
# dataset 's3://eu-customers/orders' may not be processed in region 'us-east-1': permitted in eu-north-1 (GDPR Art. 44)
```

Install the catalog once with `set_residency`, and the scheduler consults it through
`active_residency` when placing accelerator work. A deployment that installs nothing keeps an
empty `off` catalog, which permits everything.

A job reading several datasets may run only where all of them may, so
`permitted_regions` returns the intersection and `filter_regions` narrows a scheduler's
candidate list in preference order. An empty intersection is a real answer: the job has to be
split, not placed.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :toctree: generated
   :nosignatures:

   DataResidency
   ResidencyCatalog
   ResidencyVerdict
   active_residency
   set_residency

.. autodata:: RESIDENCY_MODES
```


## Entry points and query control

The calls that install a policy context and read back who is asking. {py:obj}`bt.security <batcher.security>` is a context
manager rather than a setter, because policy attaches when a table is *read*: a dataset built inside
the block stays governed for its whole life, including terminal operations that run after the block
has exited, and a table read outside every block is ungoverned.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   security
   authenticate
   set_verifier
   current_verifier
```

Two more calls govern a running *query* rather than a table, and they sit here because stopping a
query is the same kind of authority as refusing a column.
{py:func}`running_queries <batcher.running_queries>` lists the ids executing in this process, one per
terminal operation, and {py:func}`cancel_query <batcher.cancel_query>` asks one of them to stop.

Cancellation is cooperative. The engine checks the flag between morsels, between operators, and
between spill merge passes, so a query part-way through building a hash table notices when that
build finishes rather than the instant you ask.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cancel_query
   running_queries
```

Column-level lineage is not here at all. It hangs off the dataset, at
{py:obj}`ds.lineage() <batcher.Dataset.lineage>`.

## See also

- {doc}`Governance guide </user-guide/trust/governance>`: the worked introduction, with a runnable
  catalog, principal, and rewritten plan.
- {doc}`GPU fleets </user-guide/operate/running/gpu-fleets>`: residency as a placement constraint, beside the
  power and device-health controls a GPU datacenter runs on.
- {doc}`Data quality </user-guide/trust/data-quality>`: validation, which composes with this.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: reading the rewrite `enforce` produced.
- {doc}`Quality gates </cookbook/data-engineering/maintenance/quality-gates>`: failing the pipeline
  rather than the dashboard.
- {doc}`The plan IR </architecture/deep-dives/query/plan-ir>`: the tree governance rewrites.
- {doc}`Dataset API </api/relational/dataset>` and {doc}`expressions </api/relational/expressions>`: what a mask lowers to.
- {doc}`/cookbook/governance/index`: masking, PII transforms, and lineage as runnable scripts.
