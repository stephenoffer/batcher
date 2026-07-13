# Governance

Row filters, column masks, and lineage. Governance is a **plan rewrite**, not a runtime
check: {py:obj}`enforce <batcher.governance.enforce>` rewrites the `LogicalPlan` before it
executes, so a principal who may not see a column never causes that column to be read.
There is no filtering pass after the fact, and no privileged bypass to forget.

```python
from batcher.governance import Principal, SecurityCatalog, Grant, Redact, enforce
```

The [governance guide](../user-guide/governance.md) is the worked introduction; this
page is the symbol reference.

:::{important}
Because enforcement is a rewrite rather than a check, there is no execution path that can
skip it. A principal who may not read a column does not read it and then get filtered; the
column never enters the plan. That is also why a policy costs a pushed-down filter rather
than a per-row callback.
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
`XXXX1234`; the fraud team sees the number. Bind a mask to one column with
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
is deterministic, so masked values still join and group correctly;
{py:obj}`Encrypt <batcher.governance.Encrypt>` is reversible with the key;
{py:obj}`Nullify <batcher.governance.Nullify>` is not.

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

## See also

:::{seealso}
- [Governance guide](../user-guide/governance.md): the worked introduction, with a runnable
  catalog, principal, and rewritten plan.
- [Data quality](../user-guide/data-quality.md): validation, which composes with this.
- [Explain plans](../user-guide/explain-plans.md): reading the rewrite `enforce` produced.
- [Quality gates](../examples/data-engineering/quality-gates.md): failing the pipeline
  rather than the dashboard.
- [The plan IR](../deep-dives/plan-ir.md): the tree governance rewrites.
- [Dataset API](dataset.md) and [expressions](expressions.md): what a mask lowers to.
:::
