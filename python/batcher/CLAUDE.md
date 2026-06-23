# Python Control Plane — quick rules

You are editing the control plane / public API. Full detail:
`.claude/rules/python-control-plane.md` and `.claude/rules/python-quality.md`.
The non-negotiables:

- **Layer independence.** `kyber`, `carbonite`, `core` MUST NOT import one
  another. Only `api` imports all three. `plan` is neutral — it imports none of
  them. Run `just lint-layers` after any import change. (`.claude/rules/architecture.md`)
- **Core measures, Kyber decides, Carbonite protects.** Keep verbs in their lanes:
  Kyber passes don't execute or collect metrics; Core doesn't optimize; Carbonite
  doesn't rewrite plans or compute results.
- **Never touch a tuple in the hot path.** Per-row/per-batch work lives in Rust
  behind the JSON IR + zero-copy Arrow boundary. No `O(rows)` Python.
- **The JSON IR is a stable wire contract.** `to_ir()` tags must match
  `bc_ir::RelOp` / `bc_expr::Expr` serde tags exactly; change both sides together.
- **Public API hygiene.** Lazy + immutable `Dataset`, expression-first, one obvious
  way to do each thing. Curated `__all__`, full type hints, docstrings, typed
  errors (`batcher._internal.errors`). Small surface — additions are commitments.
- **Clean code.** `ruff check`/`ruff format` clean, no duplication (share via the
  neutral layers), no dead code. (`.claude/rules/python-quality.md`)
- **Bounded structure.** Module ≤500 lines, `__init__` ≤120 (re-exports only), ≤12
  files/dir. Grow "many small things" as grouped-by-family modules + a registry (new
  rule → `kyber/rules/<family>.py`; new function → `plan/functions/<family>.py`; new IO
  format → `io/formats/<fmt>.py`). Keep `Dataset`/`Expr` thin builders — breadth goes
  on `.str`/`.dt`/… accessors, not new methods or mixins.
  (`.claude/rules/maintainability.md`; `just lint-structure`)
- **Ray is scheduling only** — the data plane bypasses the object store.

Before done: `just lint-py` → `just lint-layers` → `just lint-structure` → `just build`
→ `just test-py`.
Every relational change needs a **differential test vs DuckDB**
(`.claude/rules/testing.md`). Optimizer work: the `add-kyber-optimizer-pass` skill.
