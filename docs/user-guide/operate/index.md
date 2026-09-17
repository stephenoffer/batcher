# Operate

This section covers what happens after the query is correct: making it fast, and keeping it healthy while it runs.

Batcher is built to be inspected. `explain()` shows the planned operator tree with a cardinality estimate on every line, and `explain(analyze=True)` runs the query and puts the measured rows, wall time, peak memory, spill, and backend beside each estimate. Execution records what it measured for the optimizer to consume on the next run, so a query's history informs its next plan. You don't have to guess where the time went.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Making it fast
:link: /user-guide/operate/tuning/index
:link-type: doc
Read the plan, then work the levers: memory and spill, caching, pushdown, very large tables, skewed keys, object storage, and the GPU backend.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Keeping it running
:link: /user-guide/operate/running/index
:link-type: doc
Progress and structured events, metrics a scrape loop can read, the errors you will hit, and GPU fleets whose devices come and go.
:::
::::

## Where to start

Start with the symptom. A query that is correct but slow belongs in the tuning half, and the first stop there is always the plan. A query that raises, hangs, or dies partway belongs in the running half, and troubleshooting is organized by the error you are looking at.

The table below lists every page in both halves, tuning first.

| Page | What it covers |
|---|---|
| {doc}`Performance and memory <tuning/performance>` | The levers that make a correct query fast, inside a memory envelope |
| {doc}`Caching results <tuning/caching>` | Reuse a result instead of recomputing the plan |
| {doc}`Reading query plans <tuning/explain-plans>` | The plan and the measured profile, and how to find the expensive operator |
| {doc}`Best practices <tuning/best-practices>` | Patterns for pipelines that stay fast |
| {doc}`Reading a very large table <tuning/large-tables>` | Plan-time pruning, sampled estimates, and how the work is divided |
| {doc}`Skewed keys and hostile data shapes <tuning/skew>` | Why a job that fits its budget on paper can still die, and what Batcher does about it |
| {doc}`Filter and column pushdown <tuning/pushdown>` | What the optimizer can push into the scan, and what blocks it |
| {doc}`Object storage and worker locality <tuning/object-storage>` | Read concurrency against a cloud store, planner caches, and per-worker locality |
| {doc}`Running a query on the GPU <tuning/gpu>` | Asking for the device backend, and what it declines |
| {doc}`Observability <running/observability>` | The one event channel, structured logs, the dashboard, and the metrics export |
| {doc}`The terminal <running/terminal>` | What a query prints while it runs, and the one line it leaves behind |
| {doc}`Metrics <running/metrics>` | The counters a scrape loop reads, and what each execution path reports |
| {doc}`Troubleshooting <running/troubleshooting>` | The common failures, by symptom |
| {doc}`GPU fleets <running/gpu-fleets>` | Power budgets, fabric-aware placement, device health, residency |
| {doc}`Diagnose a slow GPU stage <running/gpu-diagnosis>` | Why a GPU stage was slow, when the answer is not in the plan |
| {doc}`Running on unstable nodes <running/unstable-nodes>` | Keeping a job alive when GPUs and nodes fail underneath it |

## See also

- {doc}`/configuration/index`: the settings behind every lever on these pages.
- {doc}`/configuration/fault-tolerance`: the retry and recovery settings the running half refers to.
- {doc}`/benchmarks/index`: how the engine measures up against other engines, and how those numbers were produced.
- {doc}`/user-guide/moving-data/index`: the readers and writers whose scans most of these levers act on.

```{toctree}
:hidden:

tuning/index
running/index
```
