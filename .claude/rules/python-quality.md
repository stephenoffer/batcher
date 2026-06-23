# Rule: Python Code & Public-API Quality

The control plane is the part users read, import, and extend. It MUST read like a
single author wrote it: clean, typed, deduplicated, lint-clean, with a small and
deliberate public surface. These are hard gates, not style preferences.

## Lint & format gate (ruff)

Ruff is the single source of truth for Python lint + format. Run it on every
change to `python/`, `tests/`, or `benchmarks/`:

```
just lint-py          # ruff check  +  ruff format --check  (blocking)
ruff check --fix .    # auto-fix the mechanical findings
ruff format .         # apply formatting
```

- **`ruff check` MUST be clean** (no errors, no `# noqa` added just to silence a
  real finding — fix the cause). The configured rule set (`[tool.ruff.lint]` in
  `pyproject.toml`) includes at minimum: `E/W` (pycodestyle), `F` (pyflakes —
  unused imports/vars), `I` (import sorting), `B` (bugbear), `UP` (pyupgrade),
  `SIM` (simplify), `C4` (comprehensions), `RUF`, and `F401`/`F811` for
  dead/duplicate definitions.
- **`ruff format` MUST be applied** — never hand-format. Formatting diffs and logic
  diffs don't mix in the same hunk.
- Type hints are mandatory on public functions/methods and dataclass fields
  (`from __future__ import annotations` at the top of every module). Prefer precise
  types over `Any`; `Any` at the FFI/JSON edge is acceptable, elsewhere justify it.

## No duplication, no dead code

- **DRY across subsystems.** Before writing a helper, search for an existing one —
  shared logic belongs in the neutral layers (`plan`, `metadata`, `config`,
  `_internal`), not copy-pasted into `kyber`/`carbonite`/`core` (which can't import
  each other, so copy-paste is the *only* way to share wrongly — don't). If two
  subsystems need the same utility, lift it into a neutral module.
- **Dead code is deleted, not commented out.** Unused functions, params, imports,
  branches, and `TODO`-stubs that never ran must go. `ruff` (`F401`, `F841`, ARG)
  catches the obvious cases; for unreferenced public-ish helpers, verify with
  `ruff check` + a quick `grep`/`rg` for call sites before deleting, and remove
  them. Git history is the archive — the tree stays clean.
- **No speculative generality.** Don't add config flags, abstraction layers, or
  "extensibility hooks" with no current caller. Add the seam when the second use
  case actually arrives. One concrete implementation beats a premature framework
  (the optimizer is literally "an ordered list of passes," not a 127-rule engine —
  follow that spirit).
- **Single source of truth for constants/contracts.** IR tags, default sizes,
  error messages, and schema definitions live in exactly one place (usually
  `plan`/`config`). Don't restate a literal that already has a named home.

## Public API quality

The public surface is everything reachable from `import batcher as bt` (the
`__all__` of `python/batcher/__init__.py`, `api/`, and the documented `plan.expr_ir`
expression API). Hold it to a higher bar:

- **Curated `__all__`.** Every module that exposes public names declares `__all__`.
  Internal helpers are `_`-prefixed and excluded. Nothing leaks by accident.
- **Stable and minimal.** Adding to the public API is a commitment. Prefer the
  smallest surface that covers the use case; one obvious way to do each thing (see
  `.claude/rules/python-control-plane.md`). Don't expose internal types
  (LogicalPlan internals, IR dicts, `_native`) to users.
- **Docstrings that earn their place.** Every public class/function has a docstring:
  one-line summary, then the contract (args, returns, raised errors, laziness
  semantics). Match the existing voice — concise, declarative, "one obvious way."
  Examples in docstrings must be runnable.
- **Typed and discoverable.** Full type hints so editors/`pyright` can drive
  completion. Overloads where a method legitimately takes `str | Expr`.
- **Errors are first-class.** Raise the project's typed exceptions
  (`batcher._internal.errors`, e.g. `PlanError`) with actionable messages, not bare
  `ValueError`/`assert`. Validate user input at the API edge, fail early and clearly.

## Functions and modules

- Small, single-purpose functions; extract a named helper instead of nesting or
  duplicating a block. A function that needs a comment to explain a section usually
  wants that section as its own named function.
- Pure where possible — especially Kyber passes (plan → plan, reading shared
  analysis) and `plan` constructors. Side effects (I/O, metrics, allocation) stay
  in `core`/`carbonite`/`io`.
- Module docstrings state the module's single responsibility and its layer, the way
  the existing ones do.

## Gate before "done"

`just lint-py` (ruff check + format) clean, `just lint-layers` green, full type
hints on touched public code, no new duplication or dead code. Then the test gate
in `.claude/rules/testing.md`.
