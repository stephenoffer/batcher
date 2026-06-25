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
    cargo clippy --workspace --exclude bc-py -- -D warnings

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

# Install the git pre-commit hook that runs the structure + ruff + layer gates.
install-hooks:
    ln -sf ../../tools/git-hooks/pre-commit .git/hooks/pre-commit
    @echo "pre-commit hook installed (runs: lint-structure, ruff check, lint-layers)"

# Build the documentation site. Warnings are errors, so an orphan page or a
# broken cross-reference fails the build. Doc code examples are executed
# separately by `just test-py` (tests/docs/test_doc_examples.py), which needs the
# engine built first.
docs:
    sphinx-build -b html -E -W --keep-going docs docs/_build/html
    @echo "docs built -> docs/_build/html/index.html"

# Regenerate the architecture diagram PNGs from their Graphviz sources (needs
# graphviz: `brew install graphviz`). The PNGs are committed; rerun after editing.
diagrams:
    python docs/_static/diagrams/render.py

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
