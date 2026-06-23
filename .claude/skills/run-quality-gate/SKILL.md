---
name: run-quality-gate
description: Run the full Batcher verification gate before claiming any change is done — Rust check/test, ruff, layer-independence lint, build, Python tests, clippy, the docs build, and the competitive benchmark — with how to triage each failure class. Invoke before committing, opening a PR, or asserting that a change works.
---

# Run the quality gate

Nothing is "done" until this is green. Run the steps in order — each is cheaper than
the next and catches a different class of problem. Stop and fix at the first failure
rather than pushing through.

## The sequence

```bash
# 1. Rust compiles (fast, no PyO3 link)
just check                  # cargo check --workspace --exclude bc-py

# 2. Rust unit tests — interpreter oracle, seq==par==JIT, mergeability invariants
just test-rust              # cargo test --workspace --exclude bc-py

# 3. Rust lint + format
cargo clippy --workspace --exclude bc-py -- -D warnings
cargo fmt --all --check

# 4. Python quality — ruff lint + format
just lint-py                # ruff check . && ruff format --check .

# 5. Layer independence — kyber/carbonite/core stay separate, plan stays neutral
just lint-layers            # lint-imports --config pyproject.toml

# 6. Build the engine into the venv (links PyO3)
just build                  # maturin develop

# 7. Python tests — unit + differential (vs DuckDB) + integration + doc examples
just test-py                # pytest (incl. tests/docs: every doc code block runs)

# 8. Docs build (doc-affecting changes) — warnings are errors
just docs                   # sphinx-build -E -W: orphan pages / broken refs fail

# 9. Competitive benchmark (perf-relevant changes only)
just bench                  # correctness vs DuckDB/Polars first, then timings
just bench-tpch             # TPC-H subset (Q1, Q3, Q6, Q10)
just bench-dist             # single-node == many-partition equivalence + timing
```

`just test` runs steps 1–2, 6–7 as the CI sequence; run the lint/docs/bench steps
alongside it. For a pure-Rust change you may skip 4–5 and 8; for a pure-Python
change you still need 6 (the engine must be built) before 7. Step 8 is required
whenever you touch `docs/` or the public API the docs describe.

## Triage by failure

- **`just check` / clippy fails** → a Rust type or lint error. Fix at the source;
  don't `#[allow(...)]` a real warning. Clippy is `-D warnings` — it's a gate.
- **`just test-rust` fails** → likely the parallel or JIT path disagrees with the
  sequential interpreter (the oracle), or a mergeability invariant broke. The
  interpreter is right by definition; make the other path match it. See
  `.claude/rules/rust-engine.md`.
- **`just lint-py` fails** → ruff finding. `ruff check --fix .` and `ruff format .`
  for mechanical fixes; for real findings (unused code, duplication, complexity)
  fix the cause, don't `# noqa`. See `.claude/rules/python-quality.md`.
- **`just lint-layers` fails** → you introduced a forbidden import (a subsystem
  importing another subsystem, or `plan` importing a subsystem). Move the shared
  code into a neutral layer (`plan`/`metadata`/`config`/`_internal`). See
  `.claude/rules/architecture.md`.
- **`just build` fails** → the PyO3 boundary (`bc-py`) or maturin. Often an FFI
  signature or arrow/pyo3 version mismatch.
- **`just test-py` fails** →
  - *differential* test: Batcher disagrees with DuckDB. Batcher is wrong until
    proven otherwise — don't weaken the test. Check nulls/empties/types/IR-tag
    drift (Python `to_ir()` vs Rust `serde`).
  - *unit* test: an optimizer pass changed results (must be semantics-preserving) or
    a plan-shape assertion regressed.
  - *integration* test: distributed/spilling/adaptive path; check
    single-node==distributed equivalence and memory-pressure behavior.
  - *docs* test (`tests/docs/test_doc_examples.py`): a documented code example
    references an API that no longer exists or returns something different. Fix the
    example against the real API; do not mark it `# docs: skip` to dodge a real
    breakage (skip is only for examples needing cloud/Ray/GPU/a real model). In the
    design sections (`architecture/`, `internals/`) blocks are opt-in via
    `# docs: run`.
- **`just docs` fails** → a Sphinx warning, treated as an error. Usually an orphan
  page (not in any `toctree`) or a broken cross-reference. Add the page to its
  section `index.md` toctree, or fix the link target.
- **`just bench` / `bench-tpch` / `bench-dist` fails** → if a query refuses to time,
  correctness vs DuckDB/Polars failed — fix that first. `bench-dist` failing means
  the distributed result diverged from single-node (a mergeability bug). If a ratio
  regressed, a hot path got slower; profile and fix or justify the trade. See
  `.claude/rules/performance.md`.

## Done

All steps green (benchmark with no regression for perf-relevant work). Only then is
the change complete.
