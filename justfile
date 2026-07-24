# Batcher dev tasks.  Run `just` to list.

default:
    @just --list

# Build the Rust engine into the active venv.
build:
    maturin develop

# Optimized build (release engine) into the venv.
build-release:
    maturin develop --release

# Fast type-check of all pure-Rust crates (skips the PyO3 link).
check:
    cargo check --workspace --exclude bc-py

# Run Rust unit tests on the pure crates.
test-rust:
    cargo test --workspace --exclude bc-py

# Run the Python test suite (requires `just build` first).
test-py:
    pytest

# The deterministic suite the coverage gate measures. Excludes tests/integration:
# those Ray/adaptive-learning tests are stable on their own (`just test-py`) but
# flake under coverage instrumentation's timing, which would make the gate
# non-deterministic. They still run for correctness in `test-py`.
COV_PATHS := "tests/unit tests/differential tests/property tests/io tests/docs"

# Measure Python control-plane coverage (terminal + HTML report).
cov-py:
    pytest {{COV_PATHS}} --cov=batcher --cov-report=term-missing --cov-report=html
    @echo "html coverage -> htmlcov/index.html"

# Measure Rust data-plane coverage. One-time: `cargo install cargo-llvm-cov`.
cov-rust:
    cargo llvm-cov --workspace --exclude bc-py --summary-only

# CI coverage gate: run the deterministic suite under coverage and fail below the
# ratchet floor. The floor sits just below the achieved baseline so it blocks
# regressions; raise it as coverage grows (see docs/internals/testing-strategy.md).
cov-gate:
    pytest {{COV_PATHS}} --cov=batcher --cov-report=term-missing --cov-fail-under=62

# Everything CI runs: full correctness suite (test-py) plus the coverage gate.
test: check test-rust build test-py cov-gate

# Format + lint.
fmt:
    cargo fmt --all
    cargo clippy --workspace --exclude bc-py --all-targets -- -D warnings

# Lint + format-check the Python control plane (ruff).
lint-py:
    ruff check python tests benchmarks examples
    ruff format --check python tests benchmarks examples

# Auto-fix + format the Python control plane (ruff).
fmt-py:
    ruff check --fix python tests benchmarks examples
    ruff format python tests benchmarks examples

# Verify the layer-separation import contracts.
lint-layers:
    lint-imports --config pyproject.toml

# Structural fitness: file/dir/class size limits (keeps v1's bloat from regrowing).
lint-structure:
    python tools/lint_structure.py

# The daily agentic self-improvement loop: agents review the codebase, make narrow verified
# improvements in isolated worktrees, and leave branches + a report for review. Nothing is
# merged automatically and your working tree is never touched. See tools/agentic/README.md.
daily args="":
    python tools/agentic/daily.py {{args}}

# Same loop, read-only: find work and report it, change nothing.
daily-review:
    python tools/agentic/daily.py --reviews-only

# Exercise the loop without an agent — creates worktrees and runs every pass's gate, which
# is how you catch a broken verify command before it silently passes everything.
daily-dry:
    python tools/agentic/daily.py --dry-run

# Capture the observable surfaces (optimizer rule order, IR tags, public API, IO registry,
# FFI signatures) before a refactor that claims to preserve behavior. Diff after, and a
# silent change — a rule that moved position, a format that stopped registering — shows up
# as a diff instead of as a bug later. Use around any move-and-re-export change.
surface-save path="/tmp/batcher-surface.json":
    python tools/surface_snapshot.py --save {{path}}

# Diff the current surfaces against a saved snapshot. Exits 1 on any observable change.
surface-diff path="/tmp/batcher-surface.json":
    python tools/surface_snapshot.py --diff {{path}}

# Regenerate MAP.md — the file-level index of what every module is for. It is derived
# from each module's own docstring and each crate's manifest, so it cannot drift; run
# this after adding, moving, or re-documenting a module. `--check` runs in CI.
map:
    python tools/gen_map.py

# Copy-paste detector. The subsystems cannot import each other, so copy-paste is the only
# *wrong* way to share between them — this is what catches it.
lint-duplication:
    python tools/lint_duplication.py

# The agent-facing docs (CLAUDE.md, .claude/rules, .claude/skills) must stay TRUE: every path
# and `just` recipe they name has to exist. Guidance pointing at a file that is not there is
# worse than none — the agent invents a new home for the code instead.
lint-guardrails:
    python tools/lint_guardrails.py

# Codebase-health report: dead code, near-duplicates, swallowed errors, do-nothing bodies,
# tests that cannot fail, and ordered results asserted order-independently. A *report*, not a
# gate — every detector is a heuristic, so the output is triage. Drives the
# `audit-codebase-health` skill; run it periodically, and compare the scorecard to last time.
audit-health args="":
    python tools/audit_health.py {{args}}

# Public-API docstring style: one-line summary, `.. doctest::` examples, typeless
# Args/Returns. Needs the engine built (it introspects the live objects). The
# examples it insists on are actually executed by `just docs`.
lint-docstrings:
    python tools/lint_docstrings.py

# Install the git pre-commit hook. RUN THIS FIRST, before you write any code: it is the only
# thing that stops a commit which breaks a hard invariant. A branch once shipped with the layer
# contract broken six ways because the hook existed and was never installed.
install-hooks:
    ln -sf ../../tools/git-hooks/pre-commit .git/hooks/pre-commit
    @echo "pre-commit hook installed (lint-structure, lint-duplication, lint-guardrails, ruff, lint-layers)"

# Build the documentation site. Warnings are errors, so an orphan page or a
# broken cross-reference fails the build. The doctest builder runs first, so a
# docstring `.. doctest::` example that disagrees with the engine fails here (the
# markdown code blocks under docs/ are executed separately by `just test-py`,
# tests/docs/test_doc_examples.py). Both need the engine built first.
docs:
    sphinx-build -b doctest docs docs/_build/doctest
    sphinx-build -b html -E -W --keep-going docs docs/_build/html
    @echo "docs built -> docs/_build/html/index.html"

# Regenerate the architecture diagram PNGs from their Graphviz sources (needs
# graphviz: `brew install graphviz`). The PNGs are committed; rerun after editing.
diagrams:
    python tools/diagrams/render.py

# Run TPC-H vs the single-node lineup (batcher, duckdb, polars, pyarrow). Pass extra
# flags through, e.g. `just bench --scale 10` or `just bench --engines batcher,duckdb,spark`.
bench args="":
    python benchmarks/run.py --benchmark tpch {{args}}

# Run the full TPC-H 22-query suite (alias of `bench` for discoverability).
bench-tpch args="":
    python benchmarks/run.py --benchmark tpch {{args}}

# Run the ClickBench 43-query single-table analytics suite.
bench-clickbench args="":
    python benchmarks/run.py --benchmark clickbench {{args}}

# Run the TPC-DS subset suite.
bench-tpcds args="":
    python benchmarks/run.py --benchmark tpcds {{args}}

# Run the operator-mix (single relational ops; includes PyArrow + Ray Data).
bench-ops args="":
    python benchmarks/run.py --benchmark operators {{args}}

# Run the parquet file-layout scan suite (one table; 1 big / 132MiB / many small files).
# Re-reads its corpus from S3 per repeat, so it is excluded from `bench-all`.
bench-scan args="":
    python benchmarks/run.py --benchmark scan {{args}}

# Run the multimodal image-ingest suite (list/decode/resize JPEGs) vs Ray Data + Daft.
# Reads a JPEG corpus per repeat (opt-in, excluded from `bench-all`); scale sets the count.
bench-images args="":
    python benchmarks/run.py --benchmark images {{args}}

# Run the multi-node lineup (batcher, ray, daft) across every dataset.
bench-multi args="":
    python benchmarks/run.py --benchmark all --tier multi {{args}}

# Run every dataset on the default single-node lineup.
bench-all args="":
    python benchmarks/run.py --benchmark all {{args}}

# List every registered benchmark without running anything.
bench-list:
    python benchmarks/run.py --list

# Run the distributed single-node == many-partition equivalence benchmark.
bench-dist args="":
    python benchmarks/run.py --benchmark distributed {{args}}

# Run a standalone aux benchmark by name (distributed | optimizer | shuffle).
bench-aux which:
    python benchmarks/run.py --benchmark {{which}}
