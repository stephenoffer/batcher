# Rule: Maintainability & Long-Term Structure

A file you cannot hold in your head is a file you will break. v1 proved it: 5,236
files, a 2,951-line module, a 61-method god class, a 1,597-line `__init__.py`,
8–15-level-deep directories, 475 mixin files, and a `Dataset` smeared across 132
files. v2 is the escape — and the same rot regrows file by file unless the limits are
mechanical. They are: `just lint-structure` and the pre-commit hook decide size, not a
reviewer's patience.

These limits are **subordinate to the hard invariants** in `CLAUDE.md` and the other
rule files. If a split would cross a layer (`architecture.md`), fork the wire contract
(`python-control-plane.md`), or break the crate DAG / one-`Expr`-one-`RelOp`
(`rust-engine.md`), the invariant wins — leave the file oversized and allowlisted with
a reason instead.

## The limits (mechanically checked — `tools/lint_structure.py`)

Hard (fail the commit):

- **Python module ≤ 500 lines** (soft target 400, warns).
- **Rust file ≤ 800 lines, EXCLUDING the trailing `#[cfg(test)]` module.** Rust
  co-locates unit tests; counting them would punish good test density. The checker cuts
  at the first column-0 `#[cfg(test)]`.
- **≤ 12 files per directory** (subdirectories don't count — they're how we tame
  breadth; build artifacts don't count). At the ceiling, add a subpackage grouped by
  responsibility — do not flatten names (`expr_str_funcs.py`) to dodge it.
- **≤ 5 directory levels** under `python/batcher/` and each `crates/*/src/`. v1's
  8–15-level trees were unnavigable.
- **`__init__.py` ≤ 120 lines, re-exports only.** It is a façade, not a code dump
  (v1's was 1,597 lines). Logic lives in a named module; `__init__` imports and
  re-exports with a curated `__all__`.

Soft (warn — judgment, not a gate):

- Function > 60 lines — "a function that needs a section comment wants that section as
  a named function."
- Class with > 25 public methods — a god-class smell. **Exception:** fluent builders
  (`Expr`, `Dataset`, `GroupBy`) and `*Namespace` accessors legitimately run wide (the
  Polars pattern); breadth there is capped by *accessors*, not by splitting the class.
- `class *Mixin` — prefer composition / namespace accessors (v1 had 475 mixin files).

Genuine exceptions go in `STRUCTURE_ALLOW` in `tools/lint_structure.py` with a one-line
reason — never a scattered inline marker. The active allowlist prints on every run so
exemptions stay visible and shrink over time.

## Structure conventions

- **Package-ize, don't shim.** When a module `X.py` outgrows the limit, make it a
  package `X/` (`__init__.py` re-exports, curated `__all__`). The public import path
  `batcher.…​.X` is preserved and the parent directory's file count stays flat. Rust:
  `foo.rs` → `foo/mod.rs` + submodules.
- **"Many small things" = grouped-by-family modules + a registry.** The optimizer is
  designed for hundreds of rules; the function/format surface for dozens each. Never
  one-file-per-rule (→ thousands of files) and never a god file (→ v1). The sweet spot
  (Polars `functions/*.py`, `expr/<namespace>.py`; DuckDB rule families): **one module
  per family a user names in one breath, ~10–40 units, 200–600 lines**, discovered by a
  decorator registry or `_internal/registry.py::Registry[T]`.
- **Namespace accessors cap god-objects.** Keep `Dataset`/`Expr` thin fluent builders;
  push breadth onto `.str`/`.dt`/`.list`/`.struct`/`.json` accessor namespaces. This is
  the antidote to v1's 23-mixin `Dataset`.
- **Composition and Protocols over mixins and deep ABCs.** No new `*Mixin` inheritance
  on the public API; no class with > 2 non-Protocol base classes. Extension contracts
  are `typing.Protocol`s.
- **Split on responsibility seams, never mid-concept.** A node's dataclass + its
  `to_ir` + its validation stay together. IR tag strings live in exactly one place
  (`plan/ir_tags.py`); a file references them, never redefines them.
- **No grab-bag modules.** `utils.py` / `helpers.py` / `common.py` / `misc.py` are
  banned filenames; shared logic gets a purpose-named home in a neutral layer
  (`plan`, `metadata`, `config`, `_internal`).

## Extending a subsystem: Strategy + Registry + Context

The sanctioned way to add behavior to `core`/`kyber`/`carbonite`, modeled on Kyber
(`Rule` + `RuleRegistry` + `OptimizerContext`):

- a small **Strategy** `Protocol` = the pluggable unit (a rule, an executor, a policy);
- a **Registry** (`Registry[T]`) for discovery/selection;
- one frozen **Context** dataclass threaded through, carrying read-only inputs + a
  decision log.

Neutral data contracts the strategies exchange live in `plan/`; the Protocols,
implementations, and `*Context` live inside their subsystem. Subsystems never import
each other; `api` wires them.

## Where new X goes

- New optimizer rule → `kyber/rules/<family>.py`, registered via `@rule`.
- New scalar/agg/window function → `plan/functions/<family>.py`, surfaced through the
  `api/functions.py` façade; or a `.str`/`.dt`/… accessor in `plan/expr/namespaces.py`.
- New IO format → `io/formats/<fmt>.py`, registered as a `SourceFormat`/`SinkFormat`.
- New relational operator → Rust `bc-runtime` (mergeable) + `plan/nodes/` + the IR tag.
- New execution tier (morsel/JIT/LLVM/GPU) → a `core` `Executor` strategy, not new
  call-site branching.
- New adaptive/resource decision → a `carbonite` policy.

## Anti-speculation (from python-quality, binding here)

Add an abstraction only where ≥ 2 implementations exist or one is imminent; otherwise
define the contract and leave a documented seam. **No empty frameworks** — no registry
with one entry, no base class with one subclass, no plugin system with no plugin. One
concrete implementation beats a premature framework.

## Gate before done

`just lint-structure` clean (or the file explicitly allowlisted with a reason), the
pre-commit hook installed (`just install-hooks`), plus the existing gates in
`python-quality.md` / `testing.md`.
