"""The crate DAG points one way, and only `bc-py` links PyO3.

Two of `CLAUDE.md`'s twelve invariants, both stated as MUSTs, both mechanically checkable
from the Cargo manifests, and neither gated by anything until now:

* **#5** "The crate DAG points one way... Never an upward or sideways edge."
* **#4** "Only `bc-py` links PyO3. Every other crate `cargo test`s without a Python
  interpreter."

Neither fails a build today because neither is broken. But an upward edge does not announce
itself: `cargo build` is perfectly happy with `bc-runtime` depending on `bc-interp` right up
until someone tries to reuse the runtime without the interpreter, and the PyO3 rule breaks the
moment one crate takes a convenience dependency -- at which point `just test-rust`, which runs
`--workspace --exclude bc-py`, stops being able to build the excluded crate's dependents
without an interpreter present.

The layering is written down here rather than derived from the graph. Deriving it would make
the test self-fulfilling: any edge is "correct" if the ranks are computed from the edges. These
ranks come from `.claude/rules/rust-engine.md`, and a new crate has to be given one.

What this cannot see: a dependency added through a feature flag or a `[target.*]` block, and
`[dev-dependencies]`, which are deliberately not checked -- a test may use anything.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

_CRATES = pathlib.Path(__file__).resolve().parents[2] / "crates"

#: crate -> layer. A dependency must land on a strictly lower number. Straight from the DAG in
#: `.claude/rules/rust-engine.md`: `bc-arrow` is the root, the Arrow-free near-leaves
#: (`bc-geo`/`bc-spatial`/`bc-secrets`) and the `bc-arrow`-only ones sit above it, `bc-expr` is
#: the one `Expr`, `bc-ir` and `bc-codegen` sit beside each other on top of it (codegen
#: compiles scalar `Expr` and does **not** depend on `bc-ir`), then `bc-runtime`, `bc-interp`,
#: and `bc-py` as the second assembly point.
_LAYER: dict[str, int] = {
    "bc-arrow": 0,
    "bc-geo": 1,
    "bc-spatial": 1,
    "bc-secrets": 1,
    "bc-resource": 1,
    "bc-sketches": 1,
    "bc-transport": 1,
    "bc-io": 1,
    "bc-udf": 1,
    "bc-expr": 2,
    "bc-ir": 3,
    "bc-codegen": 3,
    "bc-runtime": 4,
    "bc-interp": 5,
    "bc-py": 6,
}


def _manifests() -> dict[str, str]:
    return {p.parent.name: p.read_text() for p in sorted(_CRATES.glob("*/Cargo.toml"))}


def _workspace_deps(manifest: str) -> set[str]:
    """The `bc-*` crates this manifest declares as real dependencies.

    `[dev-dependencies]` is excluded on purpose: a test may depend on anything, and the
    invariant is about what the *library* links.
    """
    deps: set[str] = set()
    section = None
    for line in manifest.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.strip("[]")
            continue
        if section not in {"dependencies", "build-dependencies"}:
            continue
        match = re.match(r"(bc-[a-z]+)\s*(\.|=)", stripped)
        if match:
            deps.add(match.group(1))
    return deps


def test_every_crate_has_a_declared_layer():
    """A new crate must be placed before it can be checked."""
    unplaced = sorted(set(_manifests()) - set(_LAYER))
    assert not unplaced, (
        f"{unplaced} have no entry in `_LAYER`. Decide where the crate sits in the DAG "
        "(`.claude/rules/rust-engine.md`) and record it, rather than leaving it unchecked"
    )


@pytest.mark.parametrize("crate", sorted(_LAYER))
def test_no_crate_depends_upward_or_sideways(crate):
    manifests = _manifests()
    if crate not in manifests:
        pytest.skip(f"{crate} is not in the workspace")
    for dep in sorted(_workspace_deps(manifests[crate])):
        assert dep in _LAYER, f"{crate} depends on unplaced {dep}"
        assert _LAYER[dep] < _LAYER[crate], (
            f"{crate} (layer {_LAYER[crate]}) depends on {dep} (layer {_LAYER[dep]}) -- that "
            "is an upward or sideways edge. If the type is needed in both places it belongs "
            "in the lowest crate that both can see, not in a new edge"
        )


def test_only_bc_py_links_pyo3():
    offenders = sorted(
        name
        for name, text in _manifests().items()
        if re.search(r"^pyo3\s*(\.|=)", text, re.M) and name != "bc-py"
    )
    assert not offenders, (
        f"{offenders} declare a PyO3 dependency. Only `bc-py` may: every other crate has to "
        "`cargo test` with no Python interpreter, which is what `just test-rust` "
        "(`--workspace --exclude bc-py`) relies on"
    )


def test_the_scan_reads_the_workspace():
    """Guard against a vacuous suite.

    Both assertions above are "this set is empty", and an empty parse satisfies them. The
    first version of this scan matched `bc-arrow = ...` against the whole file and reported
    every crate as having **no** dependencies at all -- fifteen clean results, measuring
    nothing. Pin that the graph is actually read: the workspace is large, and the crate at
    the top really does depend on the one at the bottom.
    """
    manifests = _manifests()
    assert len(manifests) >= 14, f"found only {len(manifests)} crates"
    assert "bc-interp" in _workspace_deps(manifests["bc-py"]), (
        "bc-py no longer appears to depend on bc-interp, which it certainly does -- the "
        "manifest parse is broken"
    )
    assert _workspace_deps(manifests["bc-arrow"]) == set(), "bc-arrow is the root of the DAG"
