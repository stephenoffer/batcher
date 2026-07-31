# Internals

This section is the design-level record of the four control-plane subsystems, written for
contributors. You don't need any of it to use Batcher.

Three sections describe the engine at three zoom levels, and they are meant to be read in
this order:

| Section | Zoom level | Read it when |
|---|---|---|
| {doc}`Architecture <../architecture/index>` | The shape of the system | You want to know how the pieces fit |
| {doc}`Deep dives <../deep-dives/index>` | One mechanism at a time | You want to know why a query behaved that way |
| Internals (this section) | One subsystem's design | You are about to change the engine |

## The layers

Each subsystem owns exactly one decision, and the layering is what keeps that true.

![Batcher's layered architecture from the User API down through the Dataset API, Logical Plan, Kyber optimizer, Physical Plan, Execution Engine, Carbonite, and optional Ray.](/_static/diagrams/layer_stack.svg)

Ray is an optional dependency used only for distributed scheduling. Single-node execution
does not require it, and even on a cluster the data plane moves Arrow batches over Arrow
Flight rather than through the Ray object store.

The verbs stay in their lanes, and most subtle bugs in this codebase are a verb crossing
one: **Core measures, Kyber decides, Carbonite protects.** A Kyber pass that collects
runtime metadata, or a Core path that makes an optimization choice, compiles and passes
its tests while quietly corrupting the feedback loop that makes plans improve across runs.

![The eight-step data flow from user code through logical plan, Kyber optimization, physical plan, execution engine, Carbonite, the Rust data plane, and collected results.](/_static/diagrams/data_flow.svg)

## The subsystem pages

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`git-branch;1.1em` Kyber
:link: kyber
:link-type: doc
The optimizer: the phased rule pipeline, cost-based physical choices, learned cardinality,
and intra-query re-optimization.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Carbonite
:link: carbonite
:link-type: doc
The resource manager: memory envelopes, the buffer pool, spill, caching, and credit-based
flow control.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Execution engine
:link: execution
:link-type: doc
Pipelines and breakers, morsel scheduling, and the three execution paths that share
operator semantics.
:::

:::{grid-item-card} {octicon}`tools;1.1em` Extending Batcher
:link: extending
:link-type: doc
One recipe per extension point: an operator, a function, an optimizer rule, an IO format.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Testing strategy
:link: testing-strategy
:link-type: doc
The two correctness oracles, the differential suite, and the gates a change has to clear.
:::
::::

## What is not on this site

The formal treatment, covering the cost models, sketch error bounds, and control-theory
stability proofs, lives at `docs/internals/mathematical_foundations.md` in the repository.
It is rendered to PDF by `internals/generate_pdf.py` rather than published as a page,
because it carries its own cross-reference scheme.

Several competitive and platform parity ledgers sit beside it in the repository and are
also excluded from the build. They are working records for contributors deciding what to
build next, and they name open gaps in a register a published page should not carry.
`docs/conf.py` lists each one with the reason it is excluded.

## See also

- {doc}`../architecture/overview`: the two planes, the crate layout, and how a query runs.
- {doc}`../architecture/differentiators`: the design decisions these subsystems exist to
  serve, and where each one stops.
- {doc}`../deep-dives/index`: the same engine, one mechanism at a time, with worked examples.
- {doc}`../benchmarks/index`: what the design measures out at.

```{toctree}
:hidden:

kyber
carbonite
execution
extending
testing-strategy
```
