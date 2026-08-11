# Operate

Run the pipeline and understand what it did. This section splits in two: making a correct
query fast, and keeping a running job healthy.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Making it fast
:link: /user-guide/operate/tuning/index
:link-type: doc
Read the plan, then work the levers: performance and memory, caching, and the patterns that
keep a pipeline fast.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Keeping it running
:link: /user-guide/operate/running/index
:link-type: doc
Progress and logs, the errors you will hit, and staying alive on a GPU fleet whose devices
come and go.
:::
::::

## Every page in this section

| Page | What it covers |
|---|---|
| {doc}`Performance and memory <tuning/performance>` | The levers that make a correct query fast, inside a memory envelope |
| {doc}`Caching results <tuning/caching>` | Reuse a result instead of recomputing the plan |
| {doc}`Reading query plans <tuning/explain-plans>` | The plan and the measured profile, and how to find the expensive operator |
| {doc}`Best practices <tuning/best-practices>` | Patterns for pipelines that stay fast |
| {doc}`Reading a very large table <tuning/large-tables>` | Plan-time pruning, sampled estimates, and how the work is divided |
| {doc}`Observability <running/observability>` | Progress, structured logs, and the web dashboard |
| {doc}`Troubleshooting <running/troubleshooting>` | The common failures, by symptom |
| {doc}`GPU fleets <running/gpu-fleets>` | Power budgets, fabric-aware placement, device health, residency |
| {doc}`Diagnose a slow GPU stage <running/gpu-diagnosis>` | Why a GPU stage was slow, when the answer is not in the plan |
| {doc}`Running on unstable nodes <running/unstable-nodes>` | Keeping a job alive when GPUs and nodes fail underneath it |

## See also

- {doc}`/configuration/index`: the settings behind every lever on these pages.
- {doc}`/benchmarks/index`: what the engine measures out at once it is tuned.

```{toctree}
:hidden:

tuning/index
running/index
```
