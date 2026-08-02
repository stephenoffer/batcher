# Python Control Plane

You are editing the control plane / public API. This file carries what you need for
almost any Python change; the deep reference is `.claude/rules/python-control-plane.md`,
`python-quality.md`, `architecture.md`, and `maintainability.md`.

## The import matrix — check yours before you import

A package may import anything **strictly below** its line, never above or sideways.

| Layer | Package | May import |
|---|---|---|
| 5 · surface | `api` · `ml` · `graph` · `_sql` — the public API and the libraries it composes. **One layer**: they import each other by design (`ml` uses `Dataset`, `ds.ml` calls `ml`). `api` is still the only one that imports the subsystems. | everything below |
| 4 · backend | `dist` — distributed *scheduling* of the same operators | layers 0–3. **MUST NOT import `api`** (a cycle) |
| 3 · subsystems | `kyber` (decides) · `carbonite` (protects) · `core` (measures/executes) · `governance` (policy) | layers 0–2 |
| 2.5 · interop | `interop` — Arrow ↔ NumPy/torch/pandas conversion, `batch_format` | layers 0–2 |
| 2 · neutral | `io`, `observe` | layers 0–1 |
| 1 · contracts | `plan` (LogicalPlan, expr_ir, `to_ir`, ir_tags) · `metadata` | layer 0 |
| 0 · utilities | `config`, `_internal` | each other |

The four layer-3 subsystems are **mutually independent** (import-linter `independence`).
That has a consequence: **copy-paste is the only *wrong* way to share between them.** If
two need the same helper, lift it *down* into `plan`/`metadata`/`config`/`_internal` — never
paste it twice. (`_median` was pasted into `kyber` twice and `carbonite` once.)

**Keep the verbs in their lanes.** Kyber passes never execute or collect runtime metadata;
Core never makes optimization decisions; Carbonite never rewrites plans or computes results.
Breaking this compiles and passes tests while corrupting the loop that improves plans.

**Never `import batcher._native`** — always `from batcher._internal.native import engine`.
See CLAUDE.md's silent-failure guards for why this breaks all six contracts.

Run `just lint-layers` after any import change. A red contract is blocking.

## The API surface: lazy, immutable, expression-first

- A `Dataset` is a handle to a `LogicalPlan`. Every operation returns a **new** one; nothing
  mutates. No work happens until a terminal op (`collect`, `iter_batches`, `write_*`), where
  `api` sequences Kyber → Carbonite → Core.
- **Expressions, not lambdas.** Column work is `Expr`, which lowers to `bc_expr::Expr` and
  runs in Rust. User callbacks (`map_batches`) take whole Arrow batches, never rows.
- **One obvious way to do each thing.** `select` chooses/derives the full output;
  `with_columns` adds/replaces. Don't add a second spelling of an existing capability.
- **Never touch a tuple in the hot path.** No `O(rows)` Python, ever. If you are iterating
  rows or materializing element-by-element, that work belongs in Rust behind the IR.

## The JSON IR is a wire contract

Each `LogicalPlan` node's `to_ir()` returns `{"op": "<tag>", ...}`; tags and `snake_case`
names MUST exactly match `bc_ir::RelOp` / `bc_expr::Expr` serde tags. **Changing the IR is a
two-sided change in one commit** — Python `to_ir()` *and* the Rust `serde` definitions —
plus a differential test exercising the new shape. Drift here is a silent correctness bug.
Tags live only in `plan/ir_tags.py`; never redefine one. Lowering stays in `plan` (neutral).

Data crosses FFI **zero-copy** as pyarrow `RecordBatch` — never convert to Python
lists/dicts to move data. Narrow numeric types are normalized once at the boundary
(Int8/16/32 → Int64, Float16/32 → Float64); rely on that, don't re-coerce upstream.

**Ray is scheduling only.** Bulk Arrow batches move via `bc-transport` (Arrow Flight), never
through the Ray object store. `dist/` composes the *same* mergeable primitives as single-node
— a result MUST be identical whether produced on one node or a hundred: same row multiset,
same column names, same column *types*. The one stated exception is float reassociation (IEEE
addition is not associative, so partition count moves the last bits); see
`.claude/rules/python-control-plane.md`. Do not restate this as bit-identity — it is not what
the code does, and `assert_same` cannot see the difference either way.

## Quality gates

- **ruff is the source of truth.** `just lint-py` (check + format) must be clean. Never add a
  `# noqa` to silence a real finding. `from __future__ import annotations` at the top of
  every module; full type hints on public functions/methods and dataclass fields.
- **No dead code, no duplication, no speculative generality.** Delete rather than comment out
  — git history is the archive. Don't add a config flag, abstraction, or "extensibility hook"
  with no current caller; add the seam when the second use case actually arrives.
- **Public API** (reachable from `import batcher as bt`): curated `__all__`, `_`-prefixed
  internals, typed errors from `batcher._internal.errors` (not bare `ValueError`).
  Adding a name is a commitment.
- **Google-style docstrings** on every public name — one-line summary *inline with the
  quotes*, `Args:`/`Returns:` **without types** (those live in the signature), runnable
  `Examples:` in a `.. doctest::` block. `just lint-docstrings` enforces the style; CI also
  requires every public name to be *mentioned* in `docs/`, *rendered* by Sphinx autodoc, and
  *taught* in a guide/tutorial/`examples/` script — a name only an autosummary table knows
  about is a name nobody discovers. The doctest examples are executed by `just docs`, so an
  example that lies fails the build. Use `# doctest: +SKIP` for examples needing a GPU,
  cloud store, or real model.

## Structure

Module ≤500 lines, `__init__.py` ≤120 (re-exports only), ≤12 files/dir, ≤5 levels.
**Package-ize, don't shim**: when `X.py` outgrows the limit, make it a package `X/` whose
`__init__` re-exports a curated `__all__`, preserving the public import path. Grow "many
small things" as grouped-by-family modules + a registry — never one-file-per-rule, never a
god file. Banned filenames: `utils.py`/`helpers.py`/`common.py`/`misc.py`.

Keep `Dataset`/`Expr` thin fluent builders; breadth goes on `.str`/`.dt`/`.list`/`.struct`/
`.json` accessor namespaces, not new methods or mixins.

**If a split would cross a layer or fork the wire contract, the invariant wins** — leave the
file oversized and allowlist it in `tools/lint_structure.py::STRUCTURE_ALLOW` with a reason.

Where new code goes: rule → `kyber/rules/<family>.py` · function →
`plan/functions/<family>.py` · accessor → `plan/expr_ir/namespaces/` · IO format →
`io/formats/<category>/<fmt>.py` · shared contract → `plan/`. Full routing table: `MAP.md`.

## Before done

`just lint-py` → `just lint-layers` → `just lint-structure` → `just build` → `just test-py`.
Every relational change needs a **differential test vs DuckDB** (`tests/differential/`,
`assert_same`) — and remember it is order-*independent*, so it cannot see a sort bug.
Changed IR tags? Run `just test-rust` too. Moved a file? `just map` + `just surface-diff`.
