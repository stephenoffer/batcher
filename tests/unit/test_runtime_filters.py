"""Plan-shape unit tests for the `runtime_filters` (sideways-information-passing) family.

Every rule here *deletes rows*, so each gets a **fires** test through the real `Optimizer()`
and — the part that matters more — a no-op test for each shape where the deletion would be
unsound: a `full` join (which preserves both sides), an `anti` join's left side (whose
unmatched rows *are* the answer), a non-EXACT statistic where a proof is required, a bloom
probed across domains, and an ASOF `by` key (whose nulls match each other, unlike an
equi-join's).

Result-correctness vs DuckDB lives in `tests/differential/test_diff_runtime_filters.py`.
"""

from __future__ import annotations

import struct

import pytest

import batcher as bt

# Importing the package registers its rules into DEFAULT_REGISTRY.
import batcher.kyber.rules.extra.runtime_filters as rf
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.runtime_filters.evidence import _membership_values
from batcher.plan.bloom_index import _fnv1a_64, canonical_bytes
from batcher.plan.expr_ir import Col, IsNotNull
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.logical import Filter, Join, Limit, Scan
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

RULE_NAMES = {
    "dedup_source_predicates",
    "drop_filter_conjunct_implied_by_zonemap",
    "drop_filter_disjunct_refuted_by_zonemap",
    "empty_join_from_all_null_key",
    "empty_join_from_bloom_absent_key",
    "empty_join_from_disjoint_key_values",
    "prune_asof_right_by_on_bound",
    "prune_in_list_by_bloom",
    "prune_in_list_by_zonemap",
    "prune_join_side_in_list_by_other_side_bloom",
    "prune_join_side_in_list_by_other_side_range",
    "push_in_list_across_join_keys",
    "push_is_not_null_from_asof_on_key",
    "push_is_not_null_from_join_key",
}

# `algebraic._fold_disjunction` folds an OR-of-equalities into an `InList` node only at this
# many members; below it the OR chain survives. Both spellings are exercised below.
_IN_LIST_MIN = 5


# --- helpers -----------------------------------------------------------------


def _bloom_bytes(values, num_bits: int = 512, num_hashes: int = 4) -> bytes:
    """A `BloomIndex` wire image holding `values` — the reader's decoder, run forwards.

    Mirrors `bc_sketches::BloomFilter::to_bytes` (which `plan.bloom_index` parses), so the test
    can hand the optimizer a real index without the native engine.
    """
    words = [0] * (num_bits // 64)
    for value in values:
        h = _fnv1a_64(canonical_bytes(value))
        h1, h2 = h & 0xFFFFFFFF, (h >> 32) | 1
        for i in range(num_hashes):
            pos = (h1 + i * h2) % num_bits
            words[pos // 64] |= 1 << (pos % 64)
    return struct.pack("<QI", num_bits, num_hashes) + struct.pack(f"<{len(words)}Q", *words)


def _rewrite(ds, stats=None):
    """The fully optimized *logical* plan (every phase, ENFORCE included)."""
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _filters(plan) -> list[Filter]:
    return [n for n in walk(plan) if isinstance(n, Filter)]


def _conjuncts(plan) -> list:
    out: list = []
    for node in _filters(plan):
        out += split_conjuncts(node.predicate)
    return out


def _ops(plan) -> list[str]:
    return [c.to_ir().get("op") or c.to_ir().get("e") for c in _conjuncts(plan)]


def _has_not_null(plan, name: str) -> bool:
    return any(
        isinstance(c, IsNotNull) and isinstance(c.input, Col) and c.input.name == name
        for c in _conjuncts(plan)
    )


def _member_sets(plan, name: str) -> list[frozenset]:
    """Every multi-valued membership constraint on `name`, in *either* spelling.

    `is_in` lowers to an OR-chain of equalities and folds into an `InList` node only at
    `_IN_LIST_MIN` members, so a test that looked only for `InList` would be asserting on a
    shape the optimizer rarely produces.
    """
    out = []
    for conj in _conjuncts(plan):
        values = _membership_values(conj, name)
        if values is not None and len(values) > 1:
            out.append(frozenset(values))
    return out


def _join(plan) -> Join:
    return next(n for n in walk(plan) if isinstance(n, Join))


def _side(join: Join, marker: str):
    """The join input carrying column `marker` — join reordering may swap left and right."""
    return join.left if marker in join.left.available_columns() else join.right


def _is_empty(plan) -> bool:
    return any(isinstance(n, Limit) and n.n == 0 for n in walk(plan))


def _fact():
    return bt.from_pydict({"k": [1, 2, 2, 3], "v": [10, 20, 30, 40]})


def _dim():
    return bt.from_pydict({"k": [1, 2, 9], "w": [5, 6, 7]})


def _kstat(rows: int = 3, **kw) -> SourceStatistics:
    return SourceStatistics(row_count=rows, columns={"k": ColumnStat(**kw)})


def test_every_rule_is_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= RULE_NAMES
    assert set(rf.__all__) == RULE_NAMES


# --- the _FILTERABLE_SIDES law ------------------------------------------------
#
# The single invariant this whole family lives or dies by. A rule that reduces a side the
# join *preserves* deletes rows from the answer, and it would do so silently. So rather than
# trust each rule's own guard, this audits the outcome: for every join type, apply every
# filter-inserting rule of the family directly and assert it left the non-reducible side
# byte-identical (`is`-identical, in fact — a rule that changed it would have rebuilt it).

# Which side each join type may reduce (the table `rules.joins` owns).
_LAW = {
    "inner": {"left", "right"},
    "semi": {"left", "right"},
    "anti": {"right"},  # the anti join's unmatched LEFT rows are the answer
    "left": {"right"},  # a preserved side keeps its unmatched rows
    "right": {"left"},
    "full": set(),  # preserves both → nothing, ever
}

# The rules that insert a filter onto a join side. `sink_runtime_filter_to_source` and
# `dedup_source_predicates` are excluded: they move/deduplicate a filter *within* one side's
# subtree and never cross the join, so the law does not constrain them.
_SIDE_REDUCERS = (
    "push_is_not_null_from_join_key",
    "push_in_list_across_join_keys",
    "prune_join_side_in_list_by_other_side_bloom",
)


@pytest.mark.parametrize("how", sorted(_LAW))
@pytest.mark.parametrize("rule_name", _SIDE_REDUCERS)
def test_no_rule_ever_reduces_a_side_the_join_preserves(how, rule_name):
    # A join whose every side carries the evidence each rule keys off: an IN list on the key
    # (for the mirror), and a bloom on the other side's key (for the bloom prune).
    bloom = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    left = _fact().filter(col("k").is_in([2, 7]))
    right = _dim().filter(col("k").is_in([1, 2]))
    ds = left.join(right, on="k", how=how)
    opt = Optimizer(sources=ds._sources, source_stats=[None, bloom])
    ctx = opt._context()
    fn = next(r.node_fn for r in DEFAULT_REGISTRY.rules() if r.name == rule_name)

    join = _join(opt.logical_rewrite(ds._plan))
    out = fn(join, ctx)
    if out is None:
        return  # the rule declined outright — trivially lawful
    for side in ("left", "right"):
        if side not in _LAW[how]:
            assert getattr(out, side) is getattr(join, side), (
                f"{rule_name} modified the {side} side of a {how} join, which preserves it"
            )


# --- push_is_not_null_from_join_key -------------------------------------------


@pytest.mark.parametrize(
    ("how", "on_fact", "on_dim"),
    [
        ("inner", True, True),
        ("semi", True, True),
        ("anti", False, True),  # an anti join's null-keyed LEFT rows ARE the answer
        ("left", False, True),  # a preserved side keeps its unmatched rows
        ("right", True, False),
        ("full", False, False),  # both sides preserved → nothing may be dropped
    ],
)
def test_is_not_null_respects_filterable_sides(how, on_fact, on_dim):
    join = _join(_rewrite(_fact().join(_dim(), on="k", how=how)))
    assert _has_not_null(_side(join, "v"), "k") is on_fact
    assert _has_not_null(_side(join, "w"), "k") is on_dim


def test_is_not_null_skipped_when_null_count_proven_zero():
    ds = _fact().join(_dim(), on="k")
    stats = [_kstat(4, null_count=0, provenance=Provenance.EXACT), _kstat(null_count=0)]
    assert not _has_not_null(_rewrite(ds, stats), "k")


def test_is_not_null_skipped_when_a_filter_hides_the_proven_zero():
    """The same skip, but with a `Filter` between the scan and the join — the non-confluence bug.

    `_may_hold_null` is read at the join, and a `Filter` below it sets `null_count` to *unknown*,
    so the evidence up there says "maybe null" for a column the scan proves is null-free. The rule
    then added `k IS NOT NULL`; pushdown sank it to the scan; `drop_filter_conjunct_implied_by_
    zonemap` read the scan's EXACT `null_count=0`, proved it a tautology and deleted it; and this
    rule, finding nothing on the spine, added it right back. PUSHDOWN then ran to its iteration cap
    on every query — 16 iterations on TPC-H q3, 24 on q5, 25 on q7 — each re-walking the whole plan
    and every expression in it, and the plan a query got depended on `fixpoint_iterations`.

    The two rules must ask the same question in the same place: `_provably_true_at_source` asks it
    at the scan, where the predicate would land.
    """
    ds = _fact().filter(col("v") > 15).join(_dim(), on="k")
    stats = [
        _kstat(4, min=1, max=3, null_count=0, provenance=Provenance.EXACT),
        _kstat(null_count=0),
    ]
    assert not _has_not_null(_rewrite(ds, stats), "k")


def test_is_not_null_on_every_key_of_a_composite_join():
    left = bt.from_pydict({"a": [1], "b": [2], "v": [3]})
    right = bt.from_pydict({"a": [1], "b": [2], "w": [4]})
    plan = _rewrite(left.join(right, on=["a", "b"]))
    assert _has_not_null(plan, "a") and _has_not_null(plan, "b")


@pytest.mark.parametrize("shape", ["project", "aggregate"])
def test_runtime_filter_reaches_the_scan(shape):
    # The filter is only worth its per-row cost if `required_predicates_per_source` can see it —
    # which it can only do when the filter sits *directly* on the `Scan`. Running in PUSHDOWN is
    # what gets it there: the shipped pushdown rewrites sink it through the projection/aggregate.
    side = (
        _fact().select(k=col("k"), v2=col("v") * 2)
        if shape == "project"
        else _fact().group_by("k").agg(total=col("v").sum())
    )
    plan = _rewrite(side.join(_dim(), on="k"))
    at_scan = [f for f in _filters(plan) if isinstance(f.input, Scan) and _has_not_null(f, "k")]
    assert at_scan, "the runtime filter never reached a scan"


@pytest.mark.parametrize("marker", ["v", "w"])
def test_runtime_filter_is_added_exactly_once_under_a_fixpoint_phase(marker):
    # PUSHDOWN iterates to a fixpoint and sinks the filter below the projection, so an idempotence
    # guard that only checked the *adjacent* filter chain would stop seeing it there and re-add it
    # on every iteration. Counted per side: both sides' keys are named `k`, so one `IS NOT NULL`
    # on each is correct, not a duplicate.
    join = _join(_rewrite(_fact().select(k=col("k"), v2=col("v") * 2).join(_dim(), on="k")))
    per_side = [c for c in _conjuncts(_side(join, marker)) if isinstance(c, IsNotNull)]
    assert len(per_side) == 1, f"{len(per_side)} copies of IS NOT NULL on the {marker} side"


# --- push_in_list_across_join_keys --------------------------------------------


def test_in_list_mirrors_onto_the_other_side():
    ds = _fact().filter(col("k").is_in([1, 2])).join(_dim(), on="k")
    join = _join(_rewrite(ds))
    assert frozenset({1, 2}) in _member_sets(_side(join, "w"), "k")


def test_in_list_not_mirrored_onto_a_full_joins_side():
    ds = _fact().filter(col("k").is_in([1, 2])).join(_dim(), on="k", how="full")
    join = _join(_rewrite(ds))
    assert not _member_sets(_side(join, "w"), "k")


def test_in_list_not_mirrored_onto_an_anti_joins_left():
    # An anti join's unmatched left rows survive, so the right's IN list must NOT reach them.
    ds = _fact().join(_dim().filter(col("k").is_in([1, 2])), on="k", how="anti")
    join = _join(_rewrite(ds))
    assert not _member_sets(_side(join, "v"), "k")


def test_in_list_too_wide_is_not_mirrored():
    ds = _fact().filter(col("k").is_in(list(range(100)))).join(_dim(), on="k")
    join = _join(_rewrite(ds))
    assert not _member_sets(_side(join, "w"), "k")


# --- prune_join_side_in_list_by_other_side_bloom ------------------------------

_DIM_RANGE = {"min": 1, "max": 9, "provenance": Provenance.EXACT}


def test_bloom_prunes_a_join_key_member_the_other_side_lacks():
    # The dimension holds {1, 2, 9}; the fact asks for 2 and 7. 7 is inside [1, 9], so min/max
    # cannot rule it out — only the bloom can.
    ds = _fact().filter(col("k").is_in([2, 7])).join(_dim(), on="k")
    dim = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    join = _join(_rewrite(ds, [None, dim]))
    assert not _member_sets(_side(join, "v"), "k")  # narrowed to the single value 2


def test_bloom_does_not_prune_a_present_member():
    ds = _fact().filter(col("k").is_in([1, 2])).join(_dim(), on="k")
    dim = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    join = _join(_rewrite(ds, [None, dim]))
    assert frozenset({1, 2}) in _member_sets(_side(join, "v"), "k")


def test_bloom_domain_mismatch_prunes_nothing():
    # A bloom built over STRINGS, probed with an int key: the domain guard must decline, because
    # a cross-domain probe reports a definitive absence for a value that IS present.
    ds = _fact().filter(col("k").is_in([2, 7])).join(_dim(), on="k")
    dim = SourceStatistics(
        row_count=3,
        columns={
            "k": ColumnStat(
                min="a", max="z", provenance=Provenance.EXACT, bloom=_bloom_bytes(["a", "z"])
            )
        },
    )
    join = _join(_rewrite(ds, [None, dim]))
    assert frozenset({2, 7}) in _member_sets(_side(join, "v"), "k")


# --- ASOF ----------------------------------------------------------------------


def _asof():
    left = bt.from_pydict({"t": [10, 20, 30], "g": [1, 1, 2], "v": [1, 2, 3]})
    right = bt.from_pydict({"t": [5, 15, 99], "g": [1, 1, 2], "w": [7, 8, 9]})
    return left, right


def _asof_stats():
    return [
        SourceStatistics(row_count=3, columns={"t": ColumnStat(min=10, max=30)}),
        SourceStatistics(row_count=3, columns={"t": ColumnStat(min=5, max=99)}),
    ]


def test_asof_pushes_is_not_null_on_the_right_on_key():
    left, right = _asof()
    assert _has_not_null(_rewrite(left.join_asof(right, on="t", by="g")), "t")


def test_asof_never_pushes_is_not_null_on_a_by_key():
    # A null `by` key MATCHES a null `by` key under the ASOF row-encoded grouping — unlike an
    # equi-join. Dropping those right rows would delete real matches.
    left, right = _asof()
    assert not _has_not_null(_rewrite(left.join_asof(right, on="t", by="g")), "g")


def test_asof_backward_bounds_the_right_on_column_from_above_only():
    left, right = _asof()
    plan = _rewrite(left.join_asof(right, on="t", by="g", direction="backward"), _asof_stats())
    ops = _ops(plan)
    assert "le" in ops  # a right row past max(left.on) can never be a backward match
    assert "ge" not in ops  # …but one *below* the range still can


def test_asof_forward_bounds_the_right_on_column_from_below_only():
    left, right = _asof()
    plan = _rewrite(left.join_asof(right, on="t", by="g", direction="forward"), _asof_stats())
    ops = _ops(plan)
    assert "ge" in ops
    assert "le" not in ops


# --- zone-map / bloom skipping inside a predicate ------------------------------


def _fact_stats(**kw):
    return [SourceStatistics(row_count=4, columns={"k": ColumnStat(**kw)})]


def test_always_true_conjunct_dropped():
    stats = _fact_stats(min=1, max=3, null_count=0, provenance=Provenance.EXACT)
    plan = _rewrite(_fact().filter((col("k") >= 0) & (col("v") > 15)), stats)
    assert "ge" not in _ops(plan)


def test_always_true_conjunct_kept_when_nulls_are_possible():
    # A NULL row is dropped by the filter, so `k >= 0` is NOT a tautology over a nullable column.
    stats = _fact_stats(min=1, max=3, provenance=Provenance.EXACT)
    plan = _rewrite(_fact().filter((col("k") >= 0) & (col("v") > 15)), stats)
    assert "ge" in _ops(plan)


def test_refuted_disjunct_dropped():
    stats = _fact_stats(min=1, max=3, provenance=Provenance.EXACT)
    plan = _rewrite(_fact().filter((col("k") < 0) | (col("v") > 15)), stats)
    assert "or" not in _ops(plan) and "gt" in _ops(plan)


def test_disjunction_kept_when_no_disjunct_is_refuted():
    stats = _fact_stats(min=1, max=3, provenance=Provenance.EXACT)
    plan = _rewrite(_fact().filter((col("k") < 3) | (col("v") > 15)), stats)
    assert "or" in _ops(plan)


def test_in_list_member_out_of_range_pruned():
    # 5+ members fold into an `InList` node, which the disjunct rule cannot see into.
    stats = _fact_stats(min=1, max=3, provenance=Provenance.EXACT)
    members = [1, 2, 3, 98, 99]
    assert len(members) >= _IN_LIST_MIN
    plan = _rewrite(_fact().filter(col("k").is_in(members)), stats)
    assert _member_sets(plan, "k") == [frozenset({1, 2, 3})]


def test_in_list_all_out_of_range_empties():
    stats = _fact_stats(min=1, max=3, provenance=Provenance.EXACT)
    assert _is_empty(_rewrite(_fact().filter(col("k").is_in([95, 96, 97, 98, 99])), stats))


def test_in_list_member_absent_from_bloom_pruned():
    # 4 lies inside [1, 9] but is absent from the column — only the bloom can prove that.
    stats = _fact_stats(
        min=1, max=9, provenance=Provenance.EXACT, bloom=_bloom_bytes([1, 2, 3, 5, 9])
    )
    plan = _rewrite(_fact().filter(col("k").is_in([1, 2, 3, 4, 5])), stats)
    assert _member_sets(plan, "k") == [frozenset({1, 2, 3, 5})]


def test_in_list_bloom_domain_mismatch_prunes_nothing():
    stats = [
        SourceStatistics(
            row_count=4,
            columns={
                "k": ColumnStat(
                    min="a", max="z", provenance=Provenance.EXACT, bloom=_bloom_bytes(["a"])
                )
            },
        )
    ]
    plan = _rewrite(_fact().filter(col("k").is_in([1, 2, 3, 4, 5])), stats)
    assert _member_sets(plan, "k") == [frozenset({1, 2, 3, 4, 5})]


# --- provably-empty joins -------------------------------------------------------


def test_all_null_key_empties_an_inner_join():
    ds = _fact().join(_dim(), on="k")
    left = _kstat(4, null_count=4, provenance=Provenance.EXACT)
    assert _is_empty(_rewrite(ds, [left, None]))


def test_all_null_key_needs_exact_provenance():
    ds = _fact().join(_dim(), on="k")
    left = _kstat(4, null_count=4, provenance=Provenance.SKETCH)
    assert not _is_empty(_rewrite(ds, [left, None]))


def test_all_null_key_does_not_empty_a_left_join():
    # The preserved side's rows survive a no-match join — emptying an input deletes the answer.
    ds = _fact().join(_dim(), on="k", how="left")
    right = _kstat(3, null_count=3, provenance=Provenance.EXACT)
    assert not _is_empty(_rewrite(ds, [None, right]))


def test_disjoint_key_value_sets_empty_the_join():
    ds = _fact().filter(col("k").is_in([1, 2])).join(_dim().filter(col("k") == 9), on="k")
    assert _is_empty(_rewrite(ds))


def test_overlapping_key_value_sets_do_not_empty_the_join():
    ds = _fact().filter(col("k").is_in([1, 2])).join(_dim().filter(col("k") == 2), on="k")
    assert not _is_empty(_rewrite(ds))


def test_disjoint_key_values_do_not_empty_an_anti_join():
    right = _dim().filter(col("k") == 9)
    ds = _fact().filter(col("k").is_in([1, 2])).join(right, on="k", how="anti")
    assert not _is_empty(_rewrite(ds))


def test_bloom_absent_key_empties_the_join():
    # The fact is pinned to k = 7; the dimension's bloom proves 7 absent — and 7 is inside its
    # [1, 9] range, so `join_disjoint_keys_to_empty` cannot see this.
    ds = _fact().filter(col("k") == 7).join(_dim(), on="k")
    dim = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    assert _is_empty(_rewrite(ds, [None, dim]))


def test_bloom_present_key_does_not_empty_the_join():
    ds = _fact().filter(col("k") == 2).join(_dim(), on="k")
    dim = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    assert not _is_empty(_rewrite(ds, [None, dim]))


def test_bloom_absent_key_does_not_empty_a_left_join():
    ds = _fact().filter(col("k") == 7).join(_dim(), on="k", how="left")
    dim = _kstat(bloom=_bloom_bytes([1, 2, 9]), **_DIM_RANGE)
    assert not _is_empty(_rewrite(ds, [None, dim]))


# --- dedup_source_predicates -----------------------------------------------------


def test_repeated_conjunct_is_dropped():
    # `merge_adjacent_filters` (PUSHDOWN) fuses these into `Filter(v > 1 AND v > 1)` *after*
    # `remove_duplicate_conjuncts` (NORMALIZE) has run for the last time.
    ds = _fact().filter(col("v") > 15).filter(col("v") > 15)
    assert _ops(_rewrite(ds)) == ["gt"]


def test_no_conjunct_is_repeated_anywhere_in_an_optimized_join_plan():
    ds = _fact().filter(col("v") > 15).join(_dim(), on="k")
    for node in _filters(_rewrite(ds)):
        keys = [str(c.to_ir()) for c in split_conjuncts(node.predicate)]
        assert len(keys) == len(set(keys))
