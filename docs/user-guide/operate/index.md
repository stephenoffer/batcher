# Operate

Run the pipeline and understand what it did: caching, plans, progress, and the fixes for a slow or failing job.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Performance
:link: /user-guide/operate/performance
:link-type: doc
The levers that make a correct query fast.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Caching
:link: /user-guide/operate/caching
:link-type: doc
Reuse a result instead of recomputing it.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Explain plans
:link: /user-guide/operate/explain-plans
:link-type: doc
Read the plan and the measured profile.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Observability
:link: /user-guide/operate/observability
:link-type: doc
Progress, structured logs, and the web dashboard.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Best practices
:link: /user-guide/operate/best-practices
:link-type: doc
Patterns for pipelines that stay fast.
:::

:::{grid-item-card} {octicon}`server;1.1em` GPU fleets
:link: /user-guide/operate/gpu-fleets
:link-type: doc
Power budgets, fabric-aware placement, device health, residency.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` GPU diagnosis
:link: /user-guide/operate/gpu-diagnosis
:link-type: doc
Why a GPU stage was slow, when the answer is not in the plan.
:::

:::{grid-item-card} {octicon}`shield;1.1em` Unstable nodes
:link: /user-guide/operate/unstable-nodes
:link-type: doc
Keeping a job alive when GPUs and nodes fail underneath it.
:::

:::{grid-item-card} {octicon}`bug;1.1em` Troubleshooting
:link: /user-guide/operate/troubleshooting
:link-type: doc
Diagnose and fix common issues.
:::
::::

```{toctree}
:hidden:

performance
caching
explain-plans
observability
gpu-fleets
gpu-diagnosis
unstable-nodes
best-practices
troubleshooting
```
