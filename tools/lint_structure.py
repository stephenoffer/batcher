#!/usr/bin/env python3
"""Structural fitness checker — keeps Batcher from regrowing v1's bloat.

v1 collapsed under 5,236 files, 2,951-line modules, a 61-method god class, a
1,597-line ``__init__.py``, and 8-15-level-deep directories. This script makes those
failure modes mechanical: it fails the commit (pre-commit hook) when a file, directory,
or class crosses a size limit, and warns on the softer smells.

Run it directly to scan the whole repo::

    python tools/lint_structure.py        # or: just lint-structure

Limits live in this file (single source of truth, mirrored by
``.claude/rules/maintainability.md``). Genuine, justified exceptions go in
``STRUCTURE_ALLOW`` with a reason — never a scattered inline marker — and the active
allowlist is printed on every run so exemptions stay visible.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

# --- Limits (mirror .claude/rules/maintainability.md) -----------------------------

PY_HARD = 500  # Python module hard ceiling (lines)
PY_SOFT = 400  # Python module soft target (warn)
RUST_HARD = 800  # Rust file ceiling, EXCLUDING the trailing #[cfg(test)] module
DIR_MAX_FILES = 12  # entries per directory (excl. __pycache__)
DIR_MAX_DEPTH = 5  # directory levels under a package/src root

# Directories exempted from the file-count cap, with a reason (the dir-level analogue of
# STRUCTURE_ALLOW). Use only when the directory is deliberately the "many small things"
# grouped-by-family pattern the maintainability rule endorses, and splitting further would
# fragment one cohesive family registry. Keyed by posix path relative to the repo root.
DIR_ALLOW: dict[str, str] = {
    "python/batcher/ml/metrics": (
        "the model-metrics family package: one module per metric family (classification, "
        "regression, ranking, clustering, calibration, fairness, tables, comparison) plus their "
        "shared helpers, kept separate so each family stays discoverable and under the line limit"
    ),
    "python/batcher/kyber/rules/extra": (
        "Kyber's extended rule families: one small module per family + a registry, the "
        "sanctioned pattern for the optimizer's large (hundreds-of-rules) rule set"
    ),
    "python/batcher/ml": (
        "OVER BUDGET AND TRACKED: 29 modules against a cap of 12. Each is a public import path "
        "users depend on (batcher.ml.selection, batcher.ml.linear, batcher.ml.feature_spec, ...), "
        "so the fix is a subpackage split that keeps those paths re-exported, not a rename. "
        "Shrink it by moving whole families down the way ml/preprocessors, ml/metrics, ml/stats, "
        "ml/tabular already are — this entry is debt, not a design"
    ),
    "python/batcher/kyber": (
        "OVER BUDGET AND TRACKED: 17 modules against a cap of 12. The learned-adaptive family "
        "(cost/cardinality/calibration/cpu_shares/learning/signature) is the natural subpackage "
        "to lift out; this entry is debt, not a design"
    ),
    "benchmarks/cluster": (
        "standalone cluster benchmark scripts, run as `python benchmarks/cluster/<x>.py` — so "
        "their shared `_ray_env` bootstrap must be a SIBLING module (only the script's own "
        "directory is on sys.path), which puts the directory at 13 by one; moving it into a "
        "subpackage would break the import that de-duplicates nine copies of the bootstrap"
    ),
}
INIT_MAX = 120  # __init__.py is a re-export shim, not a code dump
FUNC_SOFT = 60  # function length soft guideline (warn)
METHODS_SOFT = 25  # public methods per class (warn) — fluent builders excepted

BANNED_FILENAMES = {"utils.py", "helpers.py", "common.py", "misc.py"}

# Roots whose subtree is governed (and the depth origin for each).
PY_ROOT = Path("python/batcher")
BENCH_ROOT = Path("benchmarks")  # the benchmark harness holds to the same structure bar
CRATE_SRC_GLOB = "crates/*/src"

# Fluent builders / namespace accessors: breadth is the sanctioned Polars pattern
# (thin builders + .str/.dt/... accessors), so the public-method *warning* is muted
# for them. They are still bound by the hard file-size limit.
FLUENT_BUILDERS = {"Expr", "Dataset", "GroupBy", "CaseBuilder", "Reader", "Writer"}
_ACCESSOR_RE = re.compile(r"Namespace$")

# Justified, visible exemptions from the hard file-size check only: path -> reason.
# Add an entry only with a one-line reason, and only when an invariant genuinely
# blocks a split (see .claude/rules/maintainability.md) — the reason is what keeps
# the list from becoming the place oversized files go to be forgotten. Every entry
# is printed on each run so the set stays visible and shrinks over time.
STRUCTURE_ALLOW: dict[str, str] = {
    # Sat at exactly 500 lines — the ceiling — so adding a single `__slots__` entry tipped
    # it over. That entry is `__weakref__`, and it is load-bearing rather than cosmetic:
    # without it the only per-instance handle on an in-memory source is `id()`, which
    # CPython reuses the moment an object is freed, so four transient frames shared one
    # learned-statistics key and each planned from another relation's distinct counts and
    # most-common-values (see `plan/source_stats.py::source_stats_key`). The real fix is to
    # split the widening helpers (`_widen_*`, the narrow-type mapping) out of the source
    # class they sit above — a genuinely separate concern — but that is a refactor to do
    # deliberately, not as a side effect of a one-line correctness fix.
    "python/batcher/io/source/inmemory.py": "at the ceiling; one __slots__ entry tipped it — extract the _widen_* helpers next",
    # Sat at 499 lines — one under the ceiling — so wiring shuffle-output replication
    # into the reduce tipped it over. The replication logic itself was extracted to
    # `dist/shuffle_replication.py` rather than left inline; what remains is the reduce
    # driver's own recovery loop. The real fix is to extract the hierarchical combiner
    # tree (`_tree_reduce*`), which is a genuinely separate concern, but that is a wider
    # refactor of a file other agents are concurrently editing — do it deliberately, not
    # as a side effect of a feature change.
    "python/batcher/dist/flight_aggregate.py": "reduce driver + recovery loop; extract _tree_reduce* next",
    # The one Expr hierarchy: the base class plus the result nodes its own methods
    # construct (Cast/MathExpr/AggExpr/Coalesce/…). They are mutually referential, so
    # splitting across modules forces a fragile base<->subclass import cycle — the
    # one-Expr invariant (rust-engine.md) wins over the line limit here.
    "python/batcher/plan/expr_ir/core.py": "one-Expr hierarchy; split forces a base/subclass import cycle",
    # The one exception hierarchy (BatcherError + every subclass). It is a single
    # cohesive contract — the subclasses are mutually referential (`.of` factories,
    # shared `__str__`) and each is publicly exported, so the docstring gate requires a
    # per-class `Examples:` doctest. Those mandatory examples, not logic, are what carry
    # it past the line limit; the suggestion helpers already live in a sibling module
    # (`suggest.py`). Splitting the exceptions across files to satisfy the counter would
    # only scatter one contract and break `from batcher._internal.errors import <Name>`.
    "python/batcher/_internal/errors/hierarchy.py": (
        "one exception hierarchy; mandatory per-class doctests inflate it"
    ),
    # The one `Expr` enum and its `serde` wire tags. `.claude/rules/rust-engine.md` and
    # crates/CLAUDE.md name this as the seam that is never cut across: the enum and its
    # tags stay in the crate's lib.rs, so the wire contract lives in exactly one place.
    # The evaluation bodies are already extracted to `eval/`.
    "crates/bc-expr/src/lib.rs": "the one Expr enum + serde wire tags; a seam rust-engine.md forbids cutting",
    # Dataset is the canonical wide fluent builder (rust-engine/maintainability rules
    # name it as legitimately wide); its heavy method bodies are already extracted to
    # dataset/_build.py, leaving thin methods + docstrings that shouldn't be cut.
    "python/batcher/api/dataset/frame.py": "Dataset fluent builder; bodies in _build.py",
    # GroupBy is a wide fluent builder like Dataset/Expr: a class of per-reducer shortcut
    # methods (sum/mean/product/mode/array_agg/skewness/…), each a thin docstring +
    # `self._reduce(name, cols)`. The bodies are already trivial; splitting a single class
    # across modules would force a base/subclass import cycle for no benefit.
    "python/batcher/api/groupby.py": "GroupBy fluent builder; per-reducer shortcut methods",
    # The single source of truth for every tunable: ~11 frozen dataclasses whose fields
    # map 1:1 to bc_ir::EngineConfig. They are one contract meant to be read together;
    # splitting them across modules would scatter that contract and the env/file/Rust
    # wiring. Public-API docstrings (python-quality.md) push it just over the limit.
    "python/batcher/config/config.py": "single config contract; maps to bc_ir::EngineConfig",
    # The Template-Method base every file-format reader subclasses — one cohesive spine
    # (path/glob/filesystem resolution, schema caching + evolution, concurrent multi-file
    # read, streaming read-ahead, split generation). Its subclasses call up into it, so a
    # split would fan a base/subclass import web across modules for no clarity gain; the
    # per-file read primitives are already the subclasses' job. Sits just over the limit.
    "python/batcher/io/base/source.py": "file-format template base; one spine subclasses call up into",
    # The parallel executor is one cohesive `match` over every RelOp arm (filter /
    # project / aggregate / sort / join / window / …); splitting arms across files
    # would scatter the dispatch and the shared spill/admit scaffolding. Operator
    # *logic* already lives in `ops/`; this file is the scheduling shell.
    "crates/bc-interp/src/par.rs": "parallel-executor dispatch hub; per-arm split scatters scheduling",
    # The projection-pushdown rule is two exhaustive per-RelOp dispatches (`_rewrite`
    # column pruning + `_visit` source-projection analysis). Like the executor hubs it
    # grows by one small arm per relational operator; splitting the dispatch across
    # files would scatter the column-need logic that must stay consistent between them.
    "python/batcher/kyber/rules/projections.py": "projection-pushdown dispatch hub; per-operator arms",
    # One aggregate-pushdown rule family (count-distinct rewrite, eager aggregation,
    # pre-aggregate through/into joins) sharing the `_MIN_PREAGG_REDUCTION` guards and
    # estimator helpers. At the size limit; splitting across files would reorder Kyber rule
    # registration within the REWRITE/SELECTION phases (import order == run order, a
    # documented correctness hazard) — the rule-order invariant wins over the size cap.
    "python/batcher/kyber/rules/agg_pushdown.py": "agg-pushdown rule family; split reorders rules",
    # The streaming across-cores executor: shard the driving scan, one pipeline per worker
    # over a shard, combine at the breaker. The prebuild cache, shard split, per-breaker
    # combine arms, shard-count cap, and sequential fallback all share one Ctx / build-cache /
    # meter scaffold; splitting them across files would fragment one cohesive scheduling path.
    "crates/bc-interp/src/stream/parallel.rs": "streaming across-cores shard/combine executor",
    # The distributed join is one cohesive strategy module — broadcast, co-partition
    # shuffle, skew salting, bloom pruning, and the post-join aggregate all share the
    # same reducer-IR / partition / Ray-task scaffolding and the `_shuffle_join`
    # fallback the broadcast path depends on. Splitting the broadcast path into a
    # sibling forces a base<->fallback import cycle with `_shuffle_join`.
    "python/batcher/dist/executors/join.py": "distributed-join strategy hub; broadcast/shuffle split forces an import cycle",
    # Cost-based join reordering: the rule driver plus three cost-DP rebuilders
    # (exhaustive subset DP, connected-subset DP for large sparse graphs, greedy) that
    # share the same edge/leaf/schema scaffolding (`_join_plans`/`_final_projection`) —
    # one memo whose enumerators are chosen by leaf count, so splitting them scatters the
    # dispatch and duplicates that scaffolding. Tracked to shrink now that the join family
    # is a package (`rules/joins/`) with room for a sibling.
    "python/batcher/kyber/rules/joins/order.py": "join-reorder rule + the cost-DP enumerators sharing one memo/scaffolding",
    # The expression accessor namespaces: each is one bound family (`.str` / `.list`)
    # whose every public method carries a Google-style docstring with a runnable
    # `.. doctest::` example (python-quality.md). The examples — not the code — push
    # these over the limit; the methods are one cohesive accessor that the factory
    # binds as a unit, so splitting them would scatter one namespace across files.
    "python/batcher/plan/expr_ir/namespaces/strings.py": "one bound .str accessor; per-method runnable examples push it over",
    "python/batcher/plan/expr_ir/namespaces/collections.py": "one bound .list accessor; per-method runnable examples push it over",
    "python/batcher/plan/expr_ir/namespaces/temporal.py": "one bound .dt accessor; per-method runnable examples push it over",
    # The IO reader/writer namespaces: one method per format/connector, each now
    # carrying a usage example (runnable for local file formats, illustrative for
    # service-backed sinks/sources). The examples, not the thin dispatch bodies, push
    # these over; they are one cohesive `bt.read` / `ds.write` façade.
    "python/batcher/api/io_namespace/reader.py": "bt.read façade; per-format examples push it over",
    "python/batcher/api/io_namespace/writer.py": "ds.write façade; per-format examples push it over",
    # The `ds.ml` accessor: one bound ML/multimodal namespace (map_batches, infer,
    # embed, the torch loaders, download/upload) whose every public method now carries
    # a Google-style docstring — runnable for the in-memory transforms, illustrative
    # for the model/loader paths. The examples, not the thin delegating bodies, push it
    # over; the methods are one cohesive accessor bound as a unit.
    "python/batcher/api/dataset/ml.py": "ds.ml accessor; per-method examples push it over",
    # A pure re-export façade, one name per line, over eight sub-packages (preprocessors,
    # encoders, tabular, metrics, stats, llm, loader, serving). It is exactly what an
    # __init__ is supposed to be; it is only over 120 lines because the ML surface has
    # more names than 120. Collapsing the imports would hide them from editors and from
    # `just lint-docstrings`, which introspects this list.
    "python/batcher/ml/__init__.py": "ML re-export facade; one name per line over 8 subpackages",
    "python/batcher/ml/preprocessors/__init__.py": (
        "the preprocessors re-export facade curates ~45 fit/transform estimators across the "
        "scalers, encoders, imputers, text, and derived submodules, one name per line so each is "
        "discoverable"
    ),
    "python/batcher/ml/stats/__init__.py": (
        "the ml.stats re-export facade curates ~60 statistics across nine submodules (descriptive, "
        "association, robust, drift, multivariate, hypothesis, homogeneity, nonparametric, and the "
        "__all__), one name per line so each is discoverable"
    ),
    "python/batcher/ml/metrics/__init__.py": (
        "the ml.metrics re-export facade curates ~80 scoring functions across nine submodules "
        "(evaluate, ranking, clustering, regression, calibration, fairness, comparison, "
        "thresholds, tables), one name per line so each is discoverable"
    ),
    "python/batcher/plan/functions/metrics/__init__.py": (
        "the ONE metric-expression facade: it re-exports 148 metrics from the 15 leaf modules of "
        "the model/ and text/ subpackages, one name per line so the docstring linter and editors "
        "see each. The subpackages deliberately have no re-export __init__ of their own, so this "
        "is the single place the surface is declared rather than one of three"
    ),
    # The single top-level expression-function facade: it re-exports every free function
    # reachable as `bt.<name>` (string, conditional, math, aggregate, quantile, regression,
    # statistics, and the metric families), one name per line in the import block and `__all__`
    # so the docstring/coverage linters and editors see each. It is a pure re-export module (no
    # logic), and it necessarily grows one line per public function added — splitting it would
    # only fragment the single discoverable `bt.*` surface for no readability gain.
    "python/batcher/api/functions.py": "the bt.* expression-function facade; pure re-exports, one line per public name",
    # The plan-layer function facade that api/functions.py re-exports through. Same story: a
    # re-export-only __init__ that curates the SQL/DataFrame-style free functions one name per
    # line, grown just past 120 by the LLM-output-parsing family. Collapsing the import/`__all__`
    # would hide the surface the coverage linter walks.
    "python/batcher/plan/functions/__init__.py": "re-export-only function facade; one line per public name for discoverability",
    # The GPU/accelerator module: vendor-agnostic backend detection plus the per-GPU
    # zero-config *recommendation* family (`recommend_quantization` / `recommend_inference_dtype`
    # / `recommend_num_gpus` / `recommend_gpu_fraction`) and the utilization-feedback loop, all
    # sharing the same backend-probe / capability / const scaffolding. `ml/` is already 17 files
    # past the directory cap, so the recommendation family cannot move to a sibling module without
    # making that worse — the dir-size invariant wins (same case as `kyber/rules/joins/order.py`).
    # It gets a home of its own when `ml/` is finally subpackaged.
    "python/batcher/ml/gpu.py": "accelerator detect + per-GPU recommendations + feedback + autocast; ml/ already over the dir cap",
    # The distributed dispatcher: one cohesive routing hub that inspects a plan's shape
    # and sends it to the matching distributed operator (map / aggregate / join / sort /
    # distinct / window / union / asof), plus the cluster-fill + envelope sizing every
    # route shares. Like `par.rs`, splitting the arms scatters the routing it exists to
    # centralize; `executors/` is at the 12-file dir cap so a sibling can't take them.
    "python/batcher/dist/executor.py": "distributed dispatch hub; per-shape routing + sizing, executors/ at 12-file cap",
    # The terminal-op conductor: one cohesive routing hub that sequences every terminal's
    # fast-paths (metadata-answer / provably-empty short-circuit / GPU backend / blob
    # offload / distributed / adaptive re-opt / spill) before falling to plain execution.
    # Splitting the ordered fast-path chain out of `_collect` scatters the very routing it
    # exists to centralize; `terminal/` groups the sibling terminals already.
    "python/batcher/api/terminal/core.py": "terminal-op conductor; ordered fast-path routing for every terminal",
    # The distributed map/inference path: one cohesive hub over its scheduling variants
    # (stateless tasks, autoscaling actor pool, query-resident pool, streamed CPU→GPU
    # stages, map→aggregate) and the data/compute-skew-adaptive task sizing they share.
    # `executors/` is at the 12-file dir cap, so the variants can't move to a sibling.
    "python/batcher/dist/executors/map.py": "distributed map/inference hub; scheduling variants + adaptive sizing, executors/ at 12-file cap",
    # The worker-side scan driver: split reading, the read-through cache sized to a share
    # of node RAM, and the `on_read_error="skip"` broken-record accounting all share the
    # same per-worker module state (the scan cache and the skip counter), so they cannot
    # move to a sibling without threading that state through. `executors/` is at the
    # 12-file dir cap. Sat at 495 until the per-query `drain_skipped_splits` (the fix that
    # makes silent PB-scale data loss observable) tipped it three lines over.
    "python/batcher/dist/executors/scan_read.py": "worker scan driver + read-through cache + skip accounting share per-worker state; executors/ at 12-file cap",
    # The one shared Flight-shuffle worker actor: every flight_* operator (aggregate /
    # join / sort / window) drives this SAME `_FlightWorker` so they share its session
    # and lineage-recovery contract. The module docstring's whole rationale is keeping
    # it single so operators share the actor without a circular import — splitting it
    # would reintroduce exactly that cycle.
    "python/batcher/dist/flight_worker.py": "single shared _FlightWorker actor for all flight ops; split reintroduces an import cycle",
    # The cardinality/stats estimator: one cohesive `StatsEstimator` (row-count
    # estimation, column stats, per-column NDV from source + learned stats, quantiles,
    # selectivity dispatch) memoized by node identity. The arms share that per-instance
    # cache state, so splitting scatters one estimator across files for ~a dozen lines.
    "python/batcher/kyber/stats/estimator.py": "cardinality/stats estimator hub; shared per-instance caches",
    # The scalar string-function family: one cohesive `StrFunc` dispatch (`.str.*`) whose
    # ~50 arms share the same UTF-8/1-based-index/null-propagation scaffolding. It grows by
    # one small arm per function; splitting the family across files would scatter the shared
    # helpers and the single match the interpreter dispatches through — same "many small
    # things = one family module" rationale the maintainability rule prescribes.
    "crates/bc-expr/src/eval/str/mod.rs": "the one .str function-family dispatch; per-fn arms share UTF-8/index/null scaffolding",
    # The window engine's per-function evaluation: frameless / running / value / ranking
    # paths over one shared partition+order scaffold. Like the executor hubs, splitting the
    # arms scatters the frame/partition logic that must agree across paths; the runtime
    # state already lives in sibling `window_frame`/`window_partition_agg` files.
    "crates/bc-runtime/src/window.rs": "window per-function dispatch; frame/partition scaffold shared across paths",
    # The canonical shuffle/partition primitive: hash + range partitioning, the parallel
    # counting-sort scatter, and the null/NaN/−0.0 routing that EVERY hash path derives key
    # identity from. It is the one place co-partitioning is defined; splitting it risks two
    # partitioners disagreeing — the exact bug class keys.rs exists to prevent.
    "crates/bc-runtime/src/shuffle.rs": "the one partition/shuffle primitive; co-partitioning defined in one place",
    # The Tier-0 executor's operator bodies: one cohesive module of the sequential-oracle
    # implementations (sort, materialize, gather) sharing the morsel/spill scaffolding. Like
    # `par.rs`, splitting the operators scatters the shared execution helpers.
    "crates/bc-interp/src/ops/mod.rs": "Tier-0 operator bodies hub; shared morsel/spill scaffolding",
    # The join primitive hub: hash / sort-merge / radix / asof strategies over one shared
    # build/probe/gather + key-canonicalization scaffold. Splitting the strategies forces a
    # base<->strategy import cycle and scatters the key-identity logic they must share.
    "crates/bc-runtime/src/join/mod.rs": "join strategy hub; strategies share build/probe/key scaffold",
    # Group-key assignment: the one place a batch's rows are mapped to group ids, over every
    # key dtype (int/float/string/bool/multi-column) with the canonical −0.0/NaN folding. It
    # is a single per-dtype dispatch; splitting scatters the folding that grouping/shuffle
    # must agree on.
    "crates/bc-runtime/src/agg/group/assign.rs": "group-id assignment per key dtype; canonical folding in one place",
}

fails: list[str] = []
warns: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


# --- Helpers ----------------------------------------------------------------------


def rust_code_lines(text: str) -> int:
    """Line count excluding the trailing top-level ``#[cfg(test)]`` module.

    The codebase convention is one trailing unit-test module per file; counting it
    would penalize good test density. We cut at the first column-0 ``#[cfg(test)]``.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() == "#[cfg(test)]":  # top-level attribute, no indentation
            return i
    return len(lines)


def class_public_methods(node: ast.ClassDef) -> list[str]:
    out = []
    for n in node.body:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and not n.name.startswith("_"):
            out.append(n.name)
    return out


def func_length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Line count excluding the docstring — the Python analogue of `rust_code_lines`.

    The point of this check is "a function that needs a section comment wants that section
    as a named function", which is a claim about *code*. Public functions here carry a
    mandatory Google-style docstring with runnable `.. doctest::` examples
    (`python-quality.md`), so counting it penalizes exactly the documentation the docstring
    gate demands: `ds.write.__call__` read as 288 lines against 222 of code, and
    `ds.ml.map_batches` as 217 against 79. Of the 214 functions this check flagged before
    the fix, 117 were over the line on docstring alone — more than half the report was
    noise, which is how a warning list stops being read.

    Same reasoning, same shape, as the Rust file check cutting at `#[cfg(test)]`.
    """
    total = (node.end_lineno or node.lineno) - node.lineno + 1
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            total -= body[0].end_lineno - body[0].lineno + 1
    return total


# --- Per-file checks --------------------------------------------------------------


def check_python_file(path: Path) -> None:
    rel = path.as_posix()
    text = path.read_text()
    n = len(text.splitlines())

    if path.name in BANNED_FILENAMES:
        fail(f"{rel}: banned grab-bag filename '{path.name}' — use a purpose-named module")

    if path.name == "__init__.py":
        # STRUCTURE_ALLOW covers `__init__.py` too. The documented escape hatch is meant to
        # apply wherever a size limit is the wrong tool, and a facade over a package whose
        # public surface genuinely exceeds 120 names is such a case; the alternative is
        # collapsing the re-exports onto shared lines, which hides them from editors and
        # from the docstring linter that introspects the surface.
        if n > INIT_MAX and STRUCTURE_ALLOW.get(rel) is None:
            fail(f"{rel}: __init__.py is {n} lines (limit {INIT_MAX}) — re-exports only")
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            fail(f"{rel}: syntax error: {e}")
            return
        logic = [
            d.name
            for d in tree.body
            if isinstance(d, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        if logic:
            warn(f"{rel}: __init__.py defines {logic} — prefer re-exports, move logic to a module")
        return

    allow = STRUCTURE_ALLOW.get(rel)
    if n > PY_HARD and allow is None:
        fail(f"{rel}: {n} lines (limit {PY_HARD})")
    elif PY_SOFT < n <= PY_HARD and allow is None:
        warn(f"{rel}: {n} lines (soft target {PY_SOFT})")

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        fail(f"{rel}: syntax error: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name not in FLUENT_BUILDERS and not _ACCESSOR_RE.search(node.name):
                pub = class_public_methods(node)
                if len(pub) > METHODS_SOFT:
                    warn(
                        f"{rel}: class {node.name} has {len(pub)} public methods "
                        f"(soft limit {METHODS_SOFT}) — push breadth to namespace accessors"
                    )
            if node.name.endswith("Mixin"):
                warn(f"{rel}: class {node.name} is a Mixin — prefer composition/accessors")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = func_length(node)
            if length > FUNC_SOFT:
                warn(f"{rel}: function {node.name} is {length} lines (soft limit {FUNC_SOFT})")


def check_rust_file(path: Path) -> None:
    rel = path.as_posix()
    if path.name in BANNED_FILENAMES:
        fail(f"{rel}: banned grab-bag filename '{path.name}'")
    code = rust_code_lines(path.read_text())
    total = len(path.read_text().splitlines())
    allow = STRUCTURE_ALLOW.get(rel)
    if code > RUST_HARD and allow is None:
        fail(f"{rel}: {code} code lines (limit {RUST_HARD}; {total} incl. tests)")


# --- Directory checks -------------------------------------------------------------


_ARTIFACT_SUFFIXES = {".so", ".pyc", ".pyd", ".dylib"}


def check_dirs(root: Path, depth_origin: Path) -> None:
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if d.name in {"__pycache__", "target"}:
            continue
        # Count *files* a reader has to scan — subdirectories (subsystems/packages) are
        # how we tame breadth, not overcrowding. Build artifacts don't count.
        files = [
            e
            for e in d.iterdir()
            if e.is_file() and not e.is_symlink() and e.suffix not in _ARTIFACT_SUFFIXES
        ]
        if len(files) > DIR_MAX_FILES and d.as_posix() not in DIR_ALLOW:
            fail(f"{d.as_posix()}/: {len(files)} files (limit {DIR_MAX_FILES})")
        rel_depth = len(d.relative_to(depth_origin).parts)
        if rel_depth > DIR_MAX_DEPTH:
            fail(f"{d.as_posix()}/: nesting depth {rel_depth} (limit {DIR_MAX_DEPTH})")


# --- Repository root --------------------------------------------------------------

# The repository root is the first thing a reader sees, so its contents are curated
# rather than accumulated. Anything tracked there must be one of these: a top-level
# package directory, a build/config manifest, or a document. Data files are the tell
# that a doctest or a benchmark wrote into the checkout — `x` (a SQLite database),
# `late.csv`, `v.parquet`, and `~/_bt_probe_tw.csv` were all committed that way before
# this check existed, and the fix is at the source (docs/conf.py chdirs the doctest
# builder into a scratch directory), not an entry here.
ROOT_ALLOWED_SUFFIXES = {".md", ".toml", ".txt", ".lock", ".cfg", ".yaml", ".yml"}
ROOT_ALLOWED_NAMES = {"justfile", "LICENSE", "NOTICE", ".gitignore", ".gitattributes"}


def check_repo_root(tracked: list[str]) -> None:
    for name in tracked:
        if "/" in name:
            continue
        path = Path(name)
        if path.name in ROOT_ALLOWED_NAMES or path.suffix in ROOT_ALLOWED_SUFFIXES:
            continue
        fail(
            f"{name}: data/scratch file tracked at the repository root — "
            "it belongs in a package"
        )


# --- Allowlist hygiene ------------------------------------------------------------
#
# An allowlist that only ever grows is debt wearing a nice hat. Every exemption here was
# a judgement call at the time it was added, and a later refactor routinely makes it
# unnecessary without anyone noticing — the entry then reads as "this file is allowed to
# be huge" long after the file stopped being huge. So the checker audits itself: an entry
# whose target no longer exists, or no longer exceeds the limit it exempts, is reported
# for deletion.
#
# Reported, never failed. A stale exemption is untidy, not broken, and failing the commit
# on it would push the next person to add exemptions rather than remove them.


def stale_allowlist_entries() -> list[str]:
    """Allowlist keys that no longer name a file (or directory) that needs the exemption."""
    stale: list[str] = []
    for rel, _reason in STRUCTURE_ALLOW.items():
        path = Path(rel)
        if not path.exists():
            stale.append(f"{rel}: no such file")
            continue
        if path.suffix == ".rs":
            if rust_code_lines(path.read_text()) <= RUST_HARD:
                stale.append(f"{rel}: now within the {RUST_HARD}-line Rust limit")
        elif len(path.read_text().splitlines()) <= PY_HARD:
            stale.append(f"{rel}: now within the {PY_HARD}-line Python limit")

    for rel, _reason in DIR_ALLOW.items():
        path = Path(rel)
        if not path.is_dir():
            stale.append(f"{rel}/: no such directory")
            continue
        files = [
            e
            for e in path.iterdir()
            if e.is_file() and not e.is_symlink() and e.suffix not in _ARTIFACT_SUFFIXES
        ]
        if len(files) <= DIR_MAX_FILES:
            stale.append(f"{rel}/: now at {len(files)} files, within the {DIR_MAX_FILES} cap")
    return stale


# --- Main -------------------------------------------------------------------------


def _tracked_files() -> list[str]:
    """Return the git-tracked paths, or an empty list outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.splitlines()


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    check_repo_root(_tracked_files())

    for root in (PY_ROOT, BENCH_ROOT):
        if root.is_dir():
            for p in root.rglob("*.py"):
                if "__pycache__" not in p.parts:
                    check_python_file(p)
            check_dirs(root, root.parent)

    for src in sorted(Path().glob(CRATE_SRC_GLOB)):
        for p in src.rglob("*.rs"):
            if "target" not in p.parts:
                check_rust_file(p)
        check_dirs(src, src)

    if STRUCTURE_ALLOW or DIR_ALLOW:
        total = len(STRUCTURE_ALLOW) + len(DIR_ALLOW)
        print(f"structure allowlist ({total} active exemptions):")
        for path, reason in {**STRUCTURE_ALLOW, **DIR_ALLOW}.items():
            print(f"  - {path}: {reason}")
        print()

        stale = stale_allowlist_entries()
        if stale:
            print(f"stale exemptions ({len(stale)}) — delete these entries:")
            for entry in stale:
                print(f"  - {entry}")
            print()

    for w in sorted(warns):
        print(f"warn: {w}")
    for f in sorted(fails):
        print(f"FAIL: {f}")

    print()
    if fails:
        print(f"lint-structure: {len(fails)} hard violation(s), {len(warns)} warning(s)")
        return 1
    print(f"lint-structure: OK ({len(warns)} warning(s), {len(STRUCTURE_ALLOW)} allowlisted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
