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
- **Docstrings that earn their place.** Every public class/function has a docstring.
  Match the existing voice — concise, declarative, "one obvious way" — and follow the
  **Google docstring style** below. The first sentence is a one-line summary on the
  same line as the opening quotes; never make it span lines. Then optional explanatory
  paragraphs, then the typed sections. Put types in the **signature only**, never in
  the docstring. Examples are runnable and wrapped in a `.. doctest::` directive under
  an `Examples:` heading.

  This is a **gate, not a preference**: `just lint-docstrings`
  (`tools/lint_docstrings.py`) introspects the live public surface — defined once in
  `tools/public_surface.py` — and fails on a missing docstring, a multi-line first
  sentence, a missing/typed `Args:`, a missing `Returns:`, or an `Examples:` block
  that isn't a `.. doctest::`. `tests/docs/test_api_coverage.py` runs the same check
  in CI, alongside four others: every public name is *mentioned* in `docs/**/*.md`;
  every public name is *rendered* by Sphinx (an `autosummary` entry, an
  `autoclass`/`autofunction`, or the `:members:` of an autodoc'd class — a prose
  mention alone publishes no docstring); every public name is *taught* outside the
  generated reference (a user guide, tutorial, ML/configuration page, or a runnable
  script under `examples/`), because a name only an `autosummary` table knows about is
  a name nobody discovers; and the expression reference page (`docs/api/expressions.md`)
  *enumerates every `Expr` method* — the fluent builder and every accessor namespace
  (`.str`/`.dt`/`.list`/`.struct`/`.json`/`.map`/`.image`/`.audio`/`.video`) — so the
  curated reference can't fall behind the surface. The `.. doctest::` examples are then
  **executed for real** by `just docs`, so an example that lies fails the build. For an
  example that needs an external resource (a cloud store, a GPU, a real model), append
  `# doctest: +SKIP` to its `>>>` lines rather than dropping the example.

  ```python
  def ray_canonical_doc_style(param1: int, param2: str) -> bool:
      """First sentence MUST be inline with the quotes and fit on one line.

      Additional explanatory text can be added in paragraphs such as this one.
      Do not introduce multi-line first sentences.

      Examples:
          .. doctest::

              >>> # Provide code examples for key use cases, as possible.
              >>> ray_canonical_doc_style(41, "hello")
              True

              >>> # A second example.
              >>> ray_canonical_doc_style(72, "goodbye")
              False

      Args:
          param1: The first parameter. Do not include the types in the
              docstring. They should be defined only in the signature.
              Multi-line parameter docs should be indented by four spaces.
          param2: The second parameter.

      Returns:
          The return value. Do not include types here.
      """
  ```
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

`just lint-py` (ruff check + format) clean, `just lint-layers` green, and — if you
touched the public API — `just lint-docstrings` clean and `just docs` green. Full type
hints on touched public code, no new duplication or dead code. Then the test gate
in `.claude/rules/testing.md`.
