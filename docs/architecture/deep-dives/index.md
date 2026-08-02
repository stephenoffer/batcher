# Deep dives

This section explains how the engine works, one mechanism at a time: what it is for, how it
works, what it costs, and where the code lives. Each page names real files, so you can stop
reading and go look.

The engine is described at three zoom levels, all nested under the Architecture section, and
they are meant to be read in this order:

| Level | Zoom | Read it when |
|---|---|---|
| {doc}`Architecture </architecture/index>` | The shape of the system | You want to know how the pieces fit |
| Deep dives (this section) | One mechanism at a time | You want to know why a query behaved that way |
| {doc}`Internals </architecture/internals/index>` | One subsystem's design | You are about to change the engine |

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` The query, end to end
:link: /architecture/deep-dives/query/query-lifecycle
:link-type: doc
From `collect()` to Arrow and back: the plan IR, the interpreter, the JIT.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Parallelism and the operator core
:link: /architecture/deep-dives/operators/mergeable-algebra
:link-type: doc
Morsels, `partial → combine → finalize`, and the four stateful operators.
:::

:::{grid-item-card} {octicon}`database;1.1em` Memory
:link: /architecture/deep-dives/memory/arrow-memory
:link-type: doc
The one columnar contract everything speaks, and what happens when the data does not fit.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Distribution
:link: /architecture/deep-dives/distribution/shuffle-flight
:link-type: doc
The Flight shuffle, credit-based flow control, Ray scheduling, and the GPU path.
:::

:::{grid-item-card} {octicon}`graph;1.1em` The adaptive layer
:link: /architecture/deep-dives/adaptive/adaptive-reoptimization
:link-type: doc
Re-planning mid-query on measured cardinalities, plus the cross-query learned-stats loop.
:::
::::

## In this section

| Group | Pages | Covers |
|---|---|---|
| {doc}`/architecture/deep-dives/query/index` | 4 | From `collect()` to Arrow: the plan IR, the interpreter, and the JIT |
| {doc}`/architecture/deep-dives/operators/index` | 6 | Morsels, the mergeable triple, and the four stateful operators |
| {doc}`/architecture/deep-dives/memory/index` | 4 | The columnar contract, the byte account, and what happens when it does not fit |
| {doc}`/architecture/deep-dives/distribution/index` | 4 | The Flight shuffle, credit flow control, scheduling, and the GPU path |
| {doc}`/architecture/deep-dives/adaptive/index` | 4 | Re-planning mid-query, cardinality, cost, and the learned loop |

## See also

- {doc}`Architecture </architecture/index>`: the shape of the system these pages sit inside.
- {doc}`Kyber </architecture/internals/kyber>`, {doc}`Carbonite </architecture/internals/carbonite>`, {doc}`the execution engine </architecture/internals/execution>`: the three subsystems, at design level.
- `docs/architecture/internals/mathematical_foundations.md` (in the repository, not a site page): the contracts, the control theory, the sketch error bounds, and the regret proofs.
- {doc}`Performance </user-guide/operate/tuning/performance>` and {doc}`reading a plan </user-guide/operate/tuning/explain-plans>`: where a reader applies all of this.
- {doc}`Benchmarks </benchmarks/index>`: the numbers these pages keep quoting.
- {doc}`Extending the engine </architecture/internals/extending>`: what to read before you change one of these mechanisms.

```{toctree}
:hidden:

query/index
operators/index
memory/index
distribution/index
adaptive/index
```
