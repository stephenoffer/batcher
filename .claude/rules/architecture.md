# Rule: The Three-Layer Architecture

This is the load-bearing rule. Batcher's correctness *and* its adaptivity depend
on the separation below. Violating it does not produce a worse design — it
produces an unstable one (the pre-contract prototype had a 15% task-failure rate).

## Control plane (Python) vs data plane (Rust)

- **Python is the control plane.** It builds plans, optimizes them, decides
  resource bounds, and orchestrates execution. It MUST NOT touch a tuple or a row
  in the hot path. If you find yourself iterating rows, computing per-element, or
  materializing data element-by-element in Python — stop; that work belongs in a
  Rust crate behind the JSON IR / FFI boundary.
- **Rust is the data plane.** All per-row and per-batch computation lives in the
  `bc-*` crates and runs over Arrow `RecordBatch`es.
- The boundary is a **JSON plan IR** plus zero-copy Arrow batches. Python lowers a
  plan to JSON (`plan/logical.py::to_ir`), hands it to `bc_py::execute_plan`, and
  gets Arrow batches back. Nothing else crosses.

## The three subsystems (independence is enforced)

`python/batcher/` is split into three subsystems that MUST stay independent:

| Subsystem    | Responsibility                                            | May import |
|--------------|-----------------------------------------------------------|------------|
| `kyber`      | **Optimizer.** Plan → plan passes; cardinality/cost; learned stats. Decides, never executes. | `plan`, `metadata`, `config`, `_internal` |
| `carbonite`  | **Resource manager.** Buffer pool, spill, credit-based flow control, memory envelopes. | `plan`, `metadata`, `config`, `_internal`, `batcher._native` (governs the data plane: the `bc-resource` pool / `bc-transport` shuffle) |
| `core`       | **Executor.** Drives the engine via `bc_py`, runs the adaptive re-optimization loop, **measures** runtime metadata. | `plan`, `metadata`, `config`, `_internal`, `batcher._native` |

Rules:

- **`kyber`, `carbonite`, `core` MUST NOT import one another.** (import-linter
  `independence` contract.)
- **`plan` is the neutral contract layer.** It MUST NOT import `kyber`,
  `carbonite`, `core`, or `api`. Everything depends on `plan`; `plan` depends on
  no subsystem. (import-linter `forbidden` contract.)
- **`api` is the only conductor.** It is the single place allowed to import all
  three subsystems; it sequences them on a terminal operation: Kyber optimizes →
  Carbonite checks feasibility / allocates → Core executes → metadata flows back.
- `metadata`, `config`, `_internal` are shared neutral utilities.

Run `just lint-layers` after any change to Python imports. A red contract is a
blocking failure, not a warning.

## The contract loop (why the split exists)

The three subsystems form a closed feedback loop with explicit hand-offs:

1. **Kyber → Carbonite**: plans carry estimated resource bounds
   (memory/CPU/network). Kyber decides *what* to run and *how much it should cost*.
2. **Carbonite → Core**: allocation primitives (reserve memory, acquire credit,
   release) with blocking semantics. Carbonite decides *whether it is feasible*
   and *protects against OOM / cascading failure*.
3. **Core → Kyber**: execution feedback (actual cardinalities, operator times,
   peak memory) recorded into the `MetadataHub`. Core *measures*; Kyber *consumes*
   that on the next run, so plans improve the more a query runs.

**Core measures, Kyber decides, Carbonite protects.** Keep these verbs in their
lanes:
- Kyber passes never make execution happen and never collect runtime metadata.
- Core never makes optimization decisions — it executes the plan it is given and
  reports what happened.
- Carbonite never rewrites plans or computes results — it manages resources.

## Where does my logic go? (decision guide)

- Choosing an algorithm / join order / build side, pruning columns, estimating
  rows → **kyber** (as a `Pass`; see `add-kyber-optimizer-pass`).
- Deciding when to spill, how many credits, how big a buffer → **carbonite**.
- Running operators, scheduling morsels, adaptive batch sizing, collecting
  metrics → **core** (orchestration) + the Rust crates (the actual compute).
- A new relational operator or expression → the **Rust data plane** first
  (`bc-ir`/`bc-expr` → `bc-interp`/`bc-runtime`), then surfaced through `plan` +
  `api` (see `add-relational-operator`).
- A shared data structure (LogicalPlan node, expr, schema, IR tag) → **plan**.

When unsure, ask: "is this a decision (kyber), a resource concern (carbonite), or
making-it-happen (core)?" and "does it touch a row?" (if yes → Rust).

See also `.claude/rules/rust-engine.md` (the data plane) and
`.claude/rules/python-control-plane.md` (the control-plane API + IR contract).
