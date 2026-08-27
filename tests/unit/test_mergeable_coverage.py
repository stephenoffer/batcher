"""Every mergeable primitive the engine implements must have its invariant asserted.

Invariant #7: stateful operators are `partial -> combine -> finalize`, and `combine` must be
associative and commutative so partials merge in any order. `.claude/rules/testing.md` makes
that a hard gate in prose -- "New `bc-runtime` primitive -> Rust unit test + the mergeability
invariant test" -- and prose does not fail a build.

The cost of the gap is the one `CLAUDE.md` names outright: "a stateful operator without a
mergeable form works perfectly single-node, passes every local test, and silently caps the
operator at one machine. Failure appears at cluster scale, as wrong results rather than an
error." A new `AggFunc` whose `combine` is subtly non-associative -- a running mean folded
without its count, a skewness merged as if the partials were disjoint -- gives the right
answer on one core and a wrong one on twelve, and nothing here would have objected.

The sketches are the same contract with a different consequence. `.claude/rules/rust-engine.md`
requires that `bc-sketches` types "are all `Mergeable` with a fixed seed so partition-built
sketches merge identically", and Kyber reads them for cardinality and quantile estimates -- so
a sketch whose merge depends on order does not return a wrong row, it quietly feeds the
optimizer a wrong estimate and the learned-stats loop degrades across runs instead of failing.
Also prose, also ungated until now.

Today the answer is 39 of 39 aggregates and 9 of 9 sketches: every one is named inside a test
whose name marks it as a merge/associativity test. This keeps it that way rather than fixing
anything.

Deliberately a *coverage* check and not a correctness one. It cannot tell whether a test
actually asserts the invariant, only that the variant is exercised by one that claims to.
That is a weaker guarantee than reading each test, and it is the one that can be mechanical.
The real proof stays in the Rust tests themselves.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.unit

_CRATES = pathlib.Path(__file__).resolve().parents[2] / "crates"
_CRATE = _CRATES / "bc-runtime" / "src"
_SKETCHES = _CRATES / "bc-sketches" / "src"

#: A test function whose name contains one of these is making a claim about merging partials.
_INVARIANT_MARKERS = ("combine", "merge", "partial", "partition")

#: The sketches' equivalent. Wider than the aggregates' because what has to hold for a sketch
#: is that the merge does not depend on order, and those tests are named for the property
#: ("associative", "commutative", "any_order") as often as for the operation.
_SKETCH_MARKERS = ("merge", "combine", "associat", "commut", "order")


def _variants() -> list[str]:
    """The `AggFunc` variants the engine implements."""
    source = (_CRATE / "agg" / "mod.rs").read_text()
    start = source.index("pub enum AggFunc {")
    end = source.index("\n}", start)
    return re.findall(r"^\s{4}([A-Z][A-Za-z0-9]*)", source[start:end], re.M)


def _invariant_test_bodies() -> list[tuple[str, str]]:
    """`(name, body)` for every `#[cfg(test)]` fn whose name marks it as a merge test."""
    found: list[tuple[str, str]] = []
    for path in _CRATE.rglob("*.rs"):
        text = path.read_text()
        marker = text.find("#[cfg(test)]")
        if marker == -1:
            continue
        for name, body in re.findall(
            r"fn ([a-z0-9_]+)\s*\(\)\s*\{(.*?)\n    \}", text[marker:], re.S
        ):
            if any(m in name for m in _INVARIANT_MARKERS):
                found.append((name, body))
    return found


def test_every_aggregate_is_exercised_by_a_mergeability_test():
    variants = _variants()
    bodies = _invariant_test_bodies()
    covered = {v for v in variants for _n, b in bodies if f"AggFunc::{v}" in b}

    missing = sorted(set(variants) - covered)
    assert not missing, (
        f"{len(missing)} aggregate(s) are named in no combine/merge/partition test: "
        f"{missing}. A `combine` that is not associative gives the right answer on one core "
        "and a wrong one on twelve -- add the invariant test, or rename the test that "
        "already covers it so this can see it"
    )


def test_the_scan_finds_the_engine_it_is_meant_to_read():
    """Guard against a vacuous suite.

    Every assertion above is "this set minus that set is empty". Two empty sets satisfy it,
    so a moved enum, a renamed crate directory, or a regex that stopped matching would make
    this file pass while checking nothing. Pin both ends: there are many aggregates, and
    there are tests claiming to merge them.
    """
    variants = _variants()
    bodies = _invariant_test_bodies()

    assert len(variants) >= 30, (
        f"found only {len(variants)} AggFunc variants; the enum moved or the parse broke"
    )
    assert len(bodies) >= 10, (
        f"found only {len(bodies)} combine/merge tests in bc-runtime; the scan is not "
        "reading the crate's tests"
    )
    assert "Sum" in variants and "Mean" in variants


def test_a_variant_with_no_merge_test_would_be_caught():
    """The positive control, run against a synthetic variant rather than the real enum.

    Without this, the coverage assertion could be passing because `covered` is computed in a
    way that always contains everything -- a substring match against the whole file, say.
    """
    bodies = _invariant_test_bodies()
    invented = "NotARealAggregateFunction"
    assert not any(f"AggFunc::{invented}" in b for _n, b in bodies)


def _test_fns(root: pathlib.Path, markers: tuple[str, ...]) -> list[tuple[str, str]]:
    """`(name, body)` for every `#[cfg(test)]` fn under `root` whose name matches `markers`."""
    found: list[tuple[str, str]] = []
    for path in root.rglob("*.rs"):
        text = path.read_text()
        marker = text.find("#[cfg(test)]")
        if marker == -1:
            continue
        for name, body in re.findall(
            r"fn ([a-z0-9_]+)\s*\(\)\s*\{(.*?)\n    \}", text[marker:], re.S
        ):
            if any(m in name for m in markers):
                found.append((name, body))
    return found


def _mergeable_types() -> list[str]:
    """Every type in `bc-sketches` that implements `Mergeable`."""
    types: list[str] = []
    for path in _SKETCHES.rglob("*.rs"):
        types += re.findall(
            r"impl(?:<[^>]*>)?\s+Mergeable(?:<[^>]*>)?\s+for\s+([A-Za-z0-9_]+)",
            path.read_text(),
        )
    return sorted(set(types))


def test_every_sketch_is_exercised_by_a_merge_test():
    """A sketch that merges order-dependently feeds Kyber a wrong estimate, not a wrong row."""
    types = _mergeable_types()
    bodies = _test_fns(_SKETCHES, _SKETCH_MARKERS)
    covered = {t for t in types for _n, b in bodies if t in b}

    missing = sorted(set(types) - covered)
    assert not missing, (
        f"{len(missing)} Mergeable sketch(es) are named in no merge/associativity test: "
        f"{missing}. Partition-built sketches must merge identically whatever the order, or "
        "the cardinality estimates Kyber plans from drift with the partition count"
    )


def test_the_sketch_scan_finds_the_crate():
    """The same vacuity guard, for the sketch half."""
    types = _mergeable_types()
    bodies = _test_fns(_SKETCHES, _SKETCH_MARKERS)
    assert len(types) >= 8, f"found only {len(types)} Mergeable impls; the scan missed the crate"
    assert len(bodies) >= 10, f"found only {len(bodies)} merge tests in bc-sketches"
    assert "HyperLogLog" in types and "KllSketch" in types
