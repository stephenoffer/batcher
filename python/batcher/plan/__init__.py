"""`plan` — the shared plan IR and inter-layer contracts.

This package is the *neutral ground* of the architecture. Kyber (optimizer),
Carbonite (resource manager), and Core (executor) all depend on `plan`, but they
do **not** depend on each other — every cross-layer value (plans, resource
bounds, feasibility verdicts, execution feedback) is defined here as an immutable
type. The layer-separation import contract (see pyproject) enforces this.

Contents:
  - `expr_ir`  : the scalar `Expr` algebra (one representation, shared with the engine)
  - `logical`  : `LogicalPlan` nodes (what the API builds)
  - `physical` : `PhysicalPlan` / `PhysicalOp` (what Kyber emits, what Core runs)
  - `schema`   : `SchemaRef` over `pyarrow.Schema` (the single source of truth for types)
  - `resource` : `ResourceBounds`, `FeasibilityVerdict`  (Kyber ↔ Carbonite)
  - `feedback` : `OperatorFeedback`                       (Core → Kyber)
  - `ids`      : stable identifiers (`OpId`)
"""

from __future__ import annotations
