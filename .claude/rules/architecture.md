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
  plan to JSON (`plan/logical/::to_ir`), hands it to `bc_py::execute_plan`, and
  gets Arrow batches back. Nothing else crosses.

## The import matrix (every package has a layer — check yours before you import)

This table is the whole answer to "what am I allowed to import?". **Every** package under
`python/batcher/` is listed. If a package is not in this table, the table is wrong — fix it
in the same commit, don't guess.

Read it bottom-up: a package may import anything **below** its line, never above or sideways
(except where the row says so).

| Layer | Package | Responsibility | May import |
|---|---|---|---|
| 5 · surface | `api` · `ml` · `graph` · `_sql` | **One layer, mutually dependent by design.** `api` is the only *conductor* — the single place that imports all four subsystems, sequencing them on a terminal op: Kyber optimizes → Carbonite checks feasibility → Core executes → metadata flows back. `ml`/`graph`/`_sql` are the feature surfaces built on it (ML/inference/loaders; graph analytics over an edge table; the SQL parser), and they lower to the same `Dataset`/`LogicalPlan` — never a second plan or a second executor. They are **not** a layer above `api`: `ml` imports `api` 77 times and `api` imports `ml` 17 times, because `ds.ml` is a façade over `ml` while `ml` is written against `Dataset`. Stating them as stacked described a one-way dependency that does not exist. | everything below |
| 4 · backend | `dist` | Distributed **scheduling** of the same operators (Ray tasks, Arrow Flight shuffle, out-of-core spill). A *scheduling* concern, not a second semantics — it composes the same mergeable primitives. | `kyber`, `carbonite`, `core` + everything below. **MUST NOT import `api`** (that is the conductor calling *into* its own backend — a cycle). |
| 3 · subsystems | `kyber` | **Optimizer.** Plan → plan passes; cardinality/cost; learned stats. Decides, never executes. | layers 0–2 |
| 3 · subsystems | `carbonite` | **Resource manager.** Buffer pool, spill, credit-based flow control, memory envelopes. Also drives the data plane it governs (`bc-resource` pool, `bc-transport` shuffle). | layers 0–2, `_internal.native` |
| 3 · subsystems | `core` | **Executor.** Drives the engine, runs the adaptive re-optimization loop, **measures** runtime metadata. | layers 0–2, `_internal.native` |
| 3 · subsystems | `governance` | **Policy.** Row filters / column masks as a pure plan rewrite; lineage. | layers 0–2 |
| 2.5 · interop | `interop` | **Arrow ↔ framework conversion** — NumPy/PyTorch/pandas/polars/JAX, and the `batch_format` surface `map_batches` presents to a user function. Neutral: it knows about arrays, never about plans or operators. It exists as its own layer because the *executor* needs it (`core.udf` converts around every user call) and the executor must not import the user-facing `ml` package, which is where this used to live. | layers 0–2 |
| 2 · neutral IO | `io` | Sources, sinks, formats, filesystem, schema evolution. **Neutral**: it imports no subsystem, so anyone may depend on it. | layers 0–1, `_internal.native` |
| 2 · neutral sinks | `observe` | **Observability sinks**: the terminal progress reporter, the bounded activity store, and the web dashboard (`bt.start_ui()`). Consumes the event bus (`_internal.events`) that every subsystem publishes to; it reads events, never the engine. **Neutral** — it imports no subsystem (not even `io`), which is what keeps observability decoupled from the thing it observes. | layers 0–1 |
| 1 · contracts | `plan` | `LogicalPlan`/`PhysicalPlan`, `expr_ir`, schema, the JSON IR (`to_ir`), IR tags. | layer 0 |
| 1 · contracts | `metadata` | Learned stats (`MetadataHub`) — Core measures, Kyber consumes. | `plan`, layer 0 |
| 0 · utilities | `config`, `_internal` | Config/profiles; errors, registry, logging, hardware, and `_internal.native` — **the one accessor for the compiled engine**. | each other only |

The four subsystems on layer 3 are **mutually independent** — `kyber`, `carbonite`, `core`,
and `governance` MUST NOT import one another (import-linter `independence` contract). That is
deliberate, and it has a consequence you must respect: **copy-paste is the only *wrong* way to
share between them.** If two subsystems need the same helper, lift it into a neutral layer
(`plan`, `metadata`, `config`, `_internal`) — never paste it twice. (This has already happened:
`_median` was pasted into `kyber` twice and `carbonite` once.)

### Never import the compiled engine directly

Use `from batcher._internal.native import engine` — **never** `import batcher._native`. The
static import graph cannot see into a compiled extension, so a direct import gets attributed to
the *root* `batcher` package, which re-exports `api`, which imports every subsystem — forging a
phantom `core -> batcher -> api -> kyber` cycle that silently breaks the independence contract.
This is not theoretical: it is what broke all six independence directions, and the
`ignore_imports` allowlist that used to paper over it is gone precisely so it cannot come back.

Run `just lint-layers` after any change to Python imports. A red contract is a blocking
failure, not a warning.

### The matrix is a contract, not a diagram

`just lint-layers` checks the whole table above as an import-linter `layers` contract, not just
the four narrow rules that used to stand in for it. Before that contract existed, **28 upward
edges had accumulated with every gate green** — all of them function-local imports, and only
one of the 28 recorded anywhere as debt.

Those 28 are listed individually in `pyproject.toml`'s `ignore_imports`, and
`just lint-layer-debt` fails if the list grows. An exemption records an edge that predates the
contract; it is **not** a way to make a new import pass. If you find yourself wanting to add
one, the layer assignment here is what is wrong, and changing it is a design decision to argue
for rather than a line to append.

### Known debt (visible on purpose)

Started at 28 upward edges; **6 remain**.

- **Paid off (22).** Five `ml.batch_format` edges went when the Arrow-to-framework conversion
  moved down into `interop`. Seventeen `api → ml`/`_sql` edges went when the matrix itself was
  corrected: `api` and the front-ends are *mutually* dependent (`ml` imports `api` 77 times,
  `api` imports `ml` 17), so they are one layer, not two stacked ones. Calling them stacked
  made ordinary façade-to-implementation calls look like violations while describing the
  coupling as one-way, which it never was.

- **`core`/`dist` → `ml.gpu` / `ml.inference`** (6 edges, open). Execution concerns — autocast,
  the inference actor pool, device sizing — still inside the user-facing package. They belong
  in `interop`. The move is measured, not guessed: the transitive closure of the eight symbols
  involved is **28 definitions, ~785 lines**, half of `ml/gpu.py`, pulling in NVML handles, the
  autocast speed probe and backend detection. It is a bisection of the one surface with no
  automated verification (`.claude/rules/device-tier.md`), so it wants a GPU and a recorded
  `gpu_shadow_verify` run — not a blind refactor.

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
