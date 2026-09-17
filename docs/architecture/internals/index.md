# Internals

This section is the design record of Batcher's subsystems, written for contributors: how the optimizer, the resource manager and the execution engine each make their one decision, how to extend them, and how every change is proven correct. You don't need any of it to use Batcher. Read it when you are about to change the engine.

Batcher is built so that a contributor can change one part without breaking another. Each subsystem owns exactly one verb, the plan crosses to Rust as a stable JSON contract, and every execution path is checked against a reference interpreter and against DuckDB. These pages explain how each of those guarantees is kept.

## Where this section sits

The architecture is documented at three zoom levels, meant to be read in order:

| Level | Zoom | Read it when |
|---|---|---|
| {doc}`Architecture </architecture/index>` | The shape of the system | You want to know how the pieces fit |
| {doc}`Deep dives </architecture/deep-dives/index>` | One mechanism at a time | You want to know why a query behaved the way it did |
| Internals, this section | One subsystem's design | You are about to change the engine |

## The layers

Each layer owns one decision, and the layering is what keeps that true. A query flows from the user API through the logical plan and Kyber's optimization to a physical plan, which the execution engine runs within the budget Carbonite grants:

![Batcher's layered architecture from the User API down through the Dataset API, Logical Plan, Kyber optimizer, Physical Plan, Execution Engine, Carbonite, and optional Ray.](/_static/diagrams/layer_stack.svg)

Ray is an optional dependency used only for distributed scheduling, and single-node execution never loads it. Even on a cluster the data plane moves Arrow batches over Arrow Flight rather than through the Ray object store, so only small control-plane messages transit Ray.

The verbs stay in their lanes: **Core measures, Kyber decides, Carbonite protects.** Most subtle bugs in this codebase are a verb crossing one. A Kyber pass that collects runtime metadata, or a Core path that makes an optimization choice, compiles and passes its tests while quietly corrupting the feedback loop that makes plans improve across runs.

![The eight-step data flow from user code through logical plan, Kyber optimization, physical plan, execution engine, Carbonite, the Rust data plane, and collected results.](/_static/diagrams/data_flow.svg)

## The subsystem pages

Each card below covers one subsystem or one contributor workflow:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`git-branch;1.1em` Kyber
:link: /architecture/internals/kyber
:link-type: doc
The optimizer: more than 700 rules in phases, cost-based physical choices, learned cardinality, and stage-boundary re-optimization.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Carbonite
:link: /architecture/internals/carbonite
:link-type: doc
The resource manager: memory envelopes, the buffer pool, spill, and credit-based flow control.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Execution engine
:link: /architecture/internals/execution
:link-type: doc
The three execution paths that share one set of operator semantics, and answering from metadata without a scan.
:::

:::{grid-item-card} {octicon}`tools;1.1em` Extending Batcher
:link: /architecture/internals/extending
:link-type: doc
One recipe per extension point: a function, an expression node, an optimizer rule, an IO format.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Testing strategy
:link: /architecture/internals/testing-strategy
:link-type: doc
The two correctness oracles, the property-based suite, and the gates a change has to clear.
:::
::::

## What isn't on this site

The formal treatment of the cost models, sketch error bounds and control-theory stability proofs lives at `docs/architecture/internals/mathematical_foundations.md` in the repository. `docs/architecture/internals/generate_pdf.py` renders it to PDF rather than publishing it as a page, because it carries its own cross-reference scheme.

The contributor working records sit beside it in `audits/`, `parity/`, `programs/` and `rfcs/`, and `docs/conf.py` excludes each directory wholesale. They are notes for deciding what to build next, and each one carries a register of open gaps and unmeasured claims. The code-checked competitive scorecard, `competitive_architecture.md`, is excluded for the same reason.

## See also

- {doc}`/architecture/overview`: the two planes, the crate layout, and how a query runs.
- {doc}`/architecture/differentiators`: the design decisions these subsystems exist to serve, and where each one stops.
- {doc}`/architecture/deep-dives/index`: the same engine one mechanism at a time, with worked examples.
- {doc}`/benchmarks/index`: what the design measures out at.

```{toctree}
:hidden:

kyber
carbonite
execution
extending
testing-strategy
```
