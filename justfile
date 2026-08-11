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
# ratchet floor. The floor sits just below the achieved baseline so it blocks regressions;
# raise it as coverage grows (see docs/architecture/internals/testing-strategy.md).
#
# It was 62 while the suite actually reached **87%** — twenty-five points of slack, about
# 20,000 statements that could stop being covered with the gate still green. A ratchet nobody
# tightens is not a ratchet. Measured 2026-08-01 at 87% over COV_PATHS; 85 leaves two points
# for the ordering variance a randomized run introduces.
cov-gate:
    pytest {{COV_PATHS}} --cov=batcher --cov-report=term-missing --cov-fail-under=85

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

# The JSON IR wire contract: every Python tag is one Rust serde will accept, and back.
# Invariant #8 was previously reconciled only by tests that happened to name a tag.
lint-ir-contract:
    python tools/lint_ir_contract.py

# Which optimizer rules ever actually fire, over a real corpus. A registered rule can be
# dead and still green: its unit tests call the function directly, so a rule waiting for a
# shape the optimizer never builds passes everything. This runs the differential suite with
# every rule counted and names the ones nothing triggered. Diagnostic, not a gate — a
# never-fired rule may just be outside the corpus, so the output is a list to review.
rule-coverage path="tests/differential":
    python tools/rule_coverage.py {{path}}

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

# How much of the suite CI cannot reach. The PR gate installs no Ray, no torch and no GPU, so
# every test needing one skips and the run still prints green — the hole is real and the trade
# is deliberate, but it has to be *visible* or a subsystem stops being exercised with no signal
# at all. This ratchets the count: `--update` to raise it when a new test genuinely needs
# hardware, `--report` to just read the table.
lint-skips:
    python tools/lint_skips.py

# The layered-architecture contract's exemption list, as a ratchet. It may shrink, never grow —
# an exemption records an upward edge that predates the contract, and is not a way to pass a
# new one. The previous attempt at an allowlist here grew by a line per module until it
# silenced a real breakage in all six directions.
lint-layer-debt:
    python tools/lint_layer_debt.py

# Every way of switching a gate off, counted. `# noqa`, `pragma: no cover`, `type: ignore`, an
# untyped `raise`, a production `assert` — each defensible alone, and the total is the thing
# that decides whether the gates mean anything. A ratchet: it may fall, never rise.
lint-suppressions:
    python tools/lint_suppressions.py

# Kyber's rule ORDER, pinned. Registration order is run order and it is decided by the import
# graph, so re-exporting a family or splitting a module reorders rules without touching one —
# a package split once shifted 283 of 302. Re-record deliberately:
#   python tests/unit/test_kyber_rule_order.py --update
lint-rule-order:
    python -m pytest tests/unit/test_kyber_rule_order.py -q

# Tests that cannot fail: an ordered result compared order-independently, an assertion that
# is true by construction, a test that asserts nothing at all. A green gate is not a green
# light — every gate passed while a spilled `descending` sort returned unsorted data, because
# the test that should have caught it compared the result as a multiset. Unlike `audit-health`
# this IS a gate: the rules are AST-based and calibrated to zero findings on this tree.
lint-tests:
    python tools/lint_tests.py

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

# Regenerate the example-library tables in docs/examples/ from the scripts themselves.
# The prose on those pages is hand-written; the 500-row tables are not, and a list of 500
# rows maintained by hand is wrong within a week. `docs` runs this in --check mode, so a
# new example that no page claims fails the build rather than going undocumented.
example-library:
    python tools/example_library.py

# Build the documentation site. Warnings are errors, so an orphan page or a
# broken cross-reference fails the build. The doctest builder runs first, so a
# docstring `.. doctest::` example that disagrees with the engine fails here (the
# markdown code blocks under docs/ are executed separately by `just test-py`,
# tests/docs/test_doc_examples.py). Both need the engine built first.
docs:
    python tools/example_library.py --check
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

# Run the full TPC-DS 99-query suite.
bench-tpcds args="":
    python benchmarks/run.py --benchmark tpcds {{args}}

# Run the Join Order Benchmark (113 queries over the real IMDb database).
# Downloads ~1.2 GiB on the first run, so it is excluded from `bench-all`.
bench-job args="":
    python benchmarks/run.py --benchmark job {{args}}

# Run the H2O.ai db-benchmark groupby task (its 10 questions).
# Defaults to the benchmark's own 1e7-row tier, so it is excluded from `bench-all`.
bench-h2o-groupby args="":
    python benchmarks/run.py --benchmark h2o-groupby {{args}}

# Run the H2O.ai db-benchmark join task (its 5 questions, three RHS sizes).
# Defaults to the benchmark's own 1e7-row tier, so it is excluded from `bench-all`.
bench-h2o-join args="":
    python benchmarks/run.py --benchmark h2o-join {{args}}

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

# Run every dataset on the default single-node lineup, except the five opt-in ones
# (scan, images, job, h2o-groupby, h2o-join) that each have their own recipe above.
bench-all args="":
    python benchmarks/run.py --benchmark all {{args}}

# List every registered benchmark without running anything.
bench-list:
    python benchmarks/run.py --list

# Prove the shuffle's key-to-reducer mapping does not depend on the CPU it was built for.
#
# The engine hashed shuffle keys with `ahash`, which picks an AES-NI backend from the
# COMPILE-TIME target_feature. Two workers built with different `-C target-cpu` therefore
# routed the same key to different reducers — splitting a GROUP BY group and dropping join
# matches, silently, on a mixed-instance cluster and nowhere else. This runs the golden
# routing vectors twice on one machine under different ISA flags: same numbers both times,
# or the portability property has been lost again.
check-hash-portability:
    cargo test -p bc-runtime --test shuffle_hash_golden
    cargo test -p bc-sketches
    RUSTFLAGS="-C target-feature=+aes" cargo test --target-dir target/aes \
        -p bc-runtime --test shuffle_hash_golden
    RUSTFLAGS="-C target-feature=+aes" cargo test --target-dir target/aes -p bc-sketches

# Run the distributed single-node == many-partition equivalence benchmark.
bench-dist args="":
    python benchmarks/run.py --benchmark distributed {{args}}

# Measure throughput and tail latency as concurrent clients are added — the axis every
# other suite here misses. A number from this is only meaningful with its four axes
# (--clients-as, --rate, --shape, and the machine fingerprint), all of which it records.
bench-qps args="":
    python benchmarks/concurrency/run.py {{args}}

# Self-test the concurrency harness on a pure-sleep workload. Needs no comparator and no
# corpus, so it runs in CI: if the harness does not scale, no engine number it produces
# means anything.
bench-qps-sanity:
    python benchmarks/concurrency/run.py --sanity

# Run a standalone aux benchmark by name (distributed | optimizer | shuffle).
bench-aux which:
    python benchmarks/run.py --benchmark {{which}}
