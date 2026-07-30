# Governance

Row filters, column masks, and lineage. Governance is a **plan rewrite**, not a runtime
check: {py:obj}`enforce <batcher.governance.enforce>` rewrites the `LogicalPlan` before it
executes, so a principal who may not see a column never causes that column to be read.
There's no filtering pass after the fact, and no privileged bypass to forget.

```python
from batcher.governance import Principal, SecurityCatalog, Grant, Redact, enforce
```

The {doc}`governance guide <../user-guide/governance>` is the worked introduction; this
page is the symbol reference.

:::{important}
Because enforcement is a rewrite rather than a check, there is no execution path that can
skip it. A principal who may not read a column doesn't read it and then get filtered. The column never enters the plan. That's also why a policy costs a pushed-down filter rather than a per-row callback.
:::

## Identity

A query runs *as* a {py:obj}`Principal <batcher.governance.Principal>`: a name, the roles
it holds, and its attributes. Attributes are what row filters compare against, so one
policy (`region = principal.attrs["region"]`) serves every user.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autoclass:: Principal
   :members:
   :no-index:

.. autoclass:: GovernanceEvent
   :members:
   :no-index:
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
JWKS, and needs the optional `pyjwt` dependency.

```{eval-rst}
.. autoclass:: batcher.governance.authn.ProcessIdentityVerifier
   :members:
   :no-index:

.. autoclass:: batcher.governance.authn.HmacTokenVerifier
   :members:
   :no-index:

.. autoclass:: batcher.governance.authn.JwtVerifier
   :members:
   :no-index:

.. autoclass:: batcher.governance.authn.CredentialVerifier
   :members:
   :no-index:
```

```{warning}
Verification is a **deployment** control, not a security boundary. Code running inside the
engine's process can construct a `Principal` with any `issuer` it likes, and no in-process
mechanism can stop it. What this buys is that a query whose identity nobody established is
refused instead of silently trusted. The boundary is still the process, so run one per
trust domain. See {doc}`../user-guide/hardening`.
```

## The catalog

{py:obj}`SecurityCatalog <batcher.governance.SecurityCatalog>` holds the policy: which
roles may select which columns, which columns are masked, and which rows each principal
may see.

```{eval-rst}
.. autoclass:: SecurityCatalog
   :members:
   :no-index:

.. autoclass:: Grant
   :members:
```

## Row filters

A row filter restricts a table to the rows a principal is allowed to see. The predicate
is evaluated against the *principal*, not the row, so it lowers into the plan as an
ordinary pushed-down filter and costs nothing extra.

```{eval-rst}
.. autoclass:: RowFilter
   :members:

.. autoclass:: MatchesAttribute
   :members:

.. autoclass:: AttributeIn
   :members:
```

## Column masks

A mask changes how a column *reads* rather than whether it reads at all. An analyst sees
`XXXX1234`, while the fraud team sees the number. Bind a mask to one column with
{py:obj}`ColumnMask <batcher.governance.ColumnMask>`, or to a *tag* with
{py:obj}`TagMask <batcher.governance.TagMask>` so it applies everywhere that tag
appears, however many tables grow later.

```{eval-rst}
.. autoclass:: ColumnMask
   :members:

.. autoclass:: TagMask
   :members:
```

### Mask functions

The masking primitives themselves. {py:obj}`Pseudonymize <batcher.governance.Pseudonymize>`
is deterministic, so masked values still join and group correctly. {py:obj}`Encrypt <batcher.governance.Encrypt>` is reversible with the key, and {py:obj}`Nullify <batcher.governance.Nullify>` isn't.

```{eval-rst}
.. autoclass:: Redact
   :members:

.. autoclass:: Nullify
   :members:

.. autoclass:: Pseudonymize
   :members:

.. autoclass:: Encrypt
   :members:
```

## Enforcement and lineage

{py:obj}`enforce <batcher.governance.enforce>` applies the catalog to a plan and reports
what it did, so the rewrite is auditable rather than invisible.
{py:obj}`column_lineage <batcher.governance.column_lineage>` traces each output column back
to the source columns it derives from. That is how a tag on a source column keeps masking a
value three transformations downstream, after it has been renamed and cast and aggregated.

```{eval-rst}
.. autofunction:: enforce

.. autofunction:: column_lineage

.. autodata:: Origin
```

## Data residency

A residency rule answers a different question from a grant: not who may read a dataset but
*where it may be computed on*. That is the half of a sovereignty obligation a scheduler can
break silently, by placing a stage in whichever region has spare accelerator capacity.

{py:obj}`ResidencyCatalog <batcher.governance.ResidencyCatalog>` holds the rules and resolves
a placement to a {py:obj}`ResidencyVerdict <batcher.governance.ResidencyVerdict>`. Its `mode`
is one of `RESIDENCY_MODES`: `off` checks nothing, `advisory` reports a refusal a caller logs
and proceeds past, and `strict` raises. An unregistered dataset is unrestricted, because
residency is an obligation you state rather than one Batcher infers from a bucket name.

```python
from batcher.governance import DataResidency, ResidencyCatalog

catalog = ResidencyCatalog(mode="strict")
catalog.register(
    DataResidency("s3://eu-customers/", frozenset({"eu-north-1"}), "GDPR Art. 44")
)

verdict = catalog.check("s3://eu-customers/orders", "us-east-1")
print(verdict.allowed)
# False
print(verdict.message())
# dataset 's3://eu-customers/orders' may not be processed in region 'us-east-1': permitted in eu-north-1 (GDPR Art. 44)
```

A job reading several datasets may run only where all of them may, so
`permitted_regions` returns the intersection and `filter_regions` narrows a scheduler's
candidate list in preference order. An empty intersection is a real answer: the job has to be
split, not placed.

```{eval-rst}
.. autoclass:: DataResidency
   :members:

.. autoclass:: ResidencyCatalog
   :members:

.. autoclass:: ResidencyVerdict
   :members:

.. autodata:: RESIDENCY_MODES
```

## See also

- {doc}`Governance guide <../user-guide/governance>`: the worked introduction, with a runnable
  catalog, principal, and rewritten plan.
- {doc}`GPU fleets <../user-guide/gpu-fleets>`: residency as a placement constraint, beside the
  power and device-health controls a GPU datacenter runs on.
- {doc}`Data quality <../user-guide/data-quality>`: validation, which composes with this.
- {doc}`Explain plans <../user-guide/explain-plans>`: reading the rewrite `enforce` produced.
- {doc}`Quality gates <../examples/data-engineering/quality-gates>`: failing the pipeline
  rather than the dashboard.
- {doc}`The plan IR <../deep-dives/plan-ir>`: the tree governance rewrites.
- {doc}`Dataset API <dataset>` and {doc}`expressions <expressions>`: what a mask lowers to.
- {doc}`../cookbook/governance/index`: masking, PII transforms, and lineage as runnable scripts.
