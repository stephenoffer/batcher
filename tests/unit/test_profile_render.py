"""The rendered shape of `explain()` — tree geometry, alignment, folding, and honesty.

`explain()` is the output a user pastes into an issue when a query is slow, so its shape
is a contract. Every test here pins a property the previous flat-indent renderer did not
have, and several of them pin a specific defect it shipped: an indent that could not be
told apart between depths, a share column measured against a denominator that made every
operator look irrelevant, and a summary that named 1% of the wall clock as the bottleneck
while saying nothing about the other 99%.
"""

from __future__ import annotations

import re

import pytest

from batcher.plan.profile import OpProfile, QueryProfile
from batcher.plan.profile.render import RenderOptions, render_profile
from batcher.plan.profile.render.tree import (
    critical_path,
    has_branch,
    last_child_flags,
    subtree_end,
    subtree_ms,
)

pytestmark = pytest.mark.unit


def _op(op_id: int, kind: str, depth: int, **kw) -> OpProfile:
    """A measured operator with sane defaults, so each test states only what it is about."""
    fields = {"measured": True, "rows_out": 10, "elapsed_ms": 1.0, "backend": "interp", **kw}
    return OpProfile(op_id=op_id, kind=kind, depth=depth, **fields)


def _join_plan() -> QueryProfile:
    """``sort → aggregate → hash_join → (filter → scan, scan)`` — the canonical shape.

    A join is the case a flat indent fails hardest on, because the reader's question is
    exactly "which side is which".
    """
    ops = (
        _op(0, "sort", 0, elapsed_ms=5.0, est_rows=100.0),
        _op(1, "aggregate", 1, elapsed_ms=200.0, est_rows=100.0),
        _op(2, "hash_join", 2, elapsed_ms=50.0, est_rows=1_000_000.0, rows_out=200),
        _op(3, "filter", 3, elapsed_ms=30.0, est_rows=200.0),
        _op(4, "scan", 4, elapsed_ms=2.0, est_rows=200.0, pushed="a > 1"),
        _op(5, "scan", 3, elapsed_ms=0.5, est_rows=10.0),
    )
    return QueryProfile(ops=ops, total_ms=400.0, rows=100, measured=True)


# --- tree geometry -----------------------------------------------------------


def test_depth_sequence_recovers_which_nodes_are_last_children():
    ops = _join_plan().ops
    # sort, aggregate, hash_join are only children; filter has a sibling scan below it.
    assert last_child_flags(ops) == [True, True, True, False, True, True]


def test_subtree_end_bounds_each_operators_descendants():
    ops = _join_plan().ops
    assert subtree_end(ops, 2) == 6  # hash_join owns filter, its scan, and the right scan
    assert subtree_end(ops, 3) == 5  # filter owns only its scan
    assert subtree_end(ops, 5) == 6  # a leaf owns nothing


def test_subtree_time_is_the_operator_plus_everything_under_it():
    ops = _join_plan().ops
    totals = subtree_ms(ops)
    assert totals[4] == pytest.approx(2.0)
    assert totals[3] == pytest.approx(32.0)
    assert totals[2] == pytest.approx(82.5)
    assert totals[0] == pytest.approx(287.5)


def test_critical_path_descends_into_the_expensive_join_side():
    """The one question a per-operator column cannot answer: which side costs the run.

    Both sides look individually modest while one of them carries the join.
    """
    ops = _join_plan().ops
    marked = critical_path(ops, subtree_ms(ops))
    assert {0, 1, 2, 3, 4} <= marked
    assert 5 not in marked, "the cheap right-hand scan is not on the hot path"


def test_a_straight_line_plan_has_no_branch_to_mark():
    chain = (_op(0, "sort", 0), _op(1, "filter", 1), _op(2, "scan", 2))
    assert has_branch(chain) is False
    assert has_branch(_join_plan().ops) is True
    # ... and the renderer therefore spends no column on a mark that would be on every row.
    assert "▶" not in render_profile(QueryProfile(ops=chain, total_ms=3.0, measured=True))


# --- the tree as drawn -------------------------------------------------------


def test_the_spine_indents_each_depth_exactly_once():
    """The defect this replaced: a depth-1 and a depth-2 node drew at the same indent.

    The root has no ancestor bar, so a depth-*d* node carries *d - 1* bar segments and then
    its own branch glyph. Off by one and every sibling pair at different depths lines up.
    """
    text = render_profile(_join_plan(), analyze=False)
    lines = [line for line in text.split("\n") if "est≈" in line]
    # One three-column bar segment per *strict* ancestor below the root, then the branch:
    # depth 1 draws flush, depth 2 carries one segment, depth 3 two, depth 4 three.
    assert any(line.startswith("└─ aggregate") for line in lines), text
    assert any(line.startswith("   └─ hash_join") for line in lines), text
    assert any(line.startswith("      │  └─ scan") for line in lines), text
    assert any(line.startswith("      └─ scan") for line in lines), text


def test_ascii_mode_draws_the_same_tree_without_box_glyphs():
    text = render_profile(_join_plan(), analyze=False, unicode=False, width=100)
    assert "`- " in text and "|- " in text and "|  " in text
    assert "└" not in text and "█" not in text


def test_estimates_stay_parseable_digits_rather_than_an_si_abbreviation():
    """Tooling parses ``est≈N``, and a reader compares it against ``actual=N``.

    ``est≈1.2M`` beside ``actual=1,203,441`` is a comparison nobody can make.
    """
    text = render_profile(_join_plan(), analyze=False)
    assert "est≈1,000,000" in text
    parsed = int(text.split("est≈")[1].split()[0].replace(",", ""))
    assert parsed == 100


def test_an_operator_says_what_it_does_beside_its_kind():
    """A plan with four joins printed four identical `hash_join` lines.

    "Which join is this one" is the first question anyone asks of a join tree, and every
    comparable engine answers it on the line (Postgres's ``Hash Cond:``, Spark's
    ``[id#3 = id#7]``, DuckDB's key list).
    """
    ops = (
        _op(0, "hash_join", 0, est_rows=10.0, detail="inner on customer"),
        _op(1, "aggregate", 1, est_rows=10.0, detail="by region · sum"),
    )
    text = render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True), analyze=False)
    assert "hash_join  [inner on customer]" in text
    assert "aggregate  [by region · sum]" in text


def test_an_operator_with_nothing_worth_naming_gets_no_empty_brackets():
    ops = (_op(0, "union", 0, est_rows=10.0),)
    assert "[]" not in render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True))


def test_an_unbudgeted_operator_prints_a_question_mark_not_a_zero():
    ops = (_op(0, "union", 0),)  # est_rows defaults to nan
    assert "est≈?" in render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True))


# --- share and accounting ----------------------------------------------------


def test_share_is_of_operator_time_so_the_column_can_rank_operators():
    """Against `total_ms` every bar on a short query is empty and ranks nothing.

    The operators here are 287.5 ms of a 400 ms wall clock; the aggregate is 70% of the
    operator time and would have read as 50% of the wall clock — a column that exists to
    rank operators against each other must not be deflated by time no operator spent.
    """
    text = render_profile(_join_plan(), analyze=True, width=140)
    aggregate = next(line for line in text.split("\n") if "aggregate" in line and "actual" in line)
    assert "70%" in aggregate


def test_the_unaccounted_remainder_is_reported_rather_than_left_to_be_inferred():
    """The 99% a short query spends outside its operators used to be invisible."""
    profile = QueryProfile(
        ops=(_op(0, "scan", 0, elapsed_ms=1.0),), total_ms=100.0, rows=10, measured=True
    )
    text = render_profile(profile, analyze=True)
    assert "where the time went" in text
    assert "operators" in text and "elsewhere" in text
    assert "99%" in text
    assert "planning, optimization, admission, FFI crossing, result assembly" in text


def test_a_parallel_operator_does_not_report_more_time_than_the_query_took():
    """`elapsed_ms` is CPU summed over workers, so dividing it by the clock printed 315%.

    Measured shape: a 20M-row filter on 64 workers reported 113ms inside a 37ms query, and
    the "elsewhere" line was clamped away by `max(0.0, total - ops_ms)` exactly when the
    operators were parallel. Both halves are asserted -- the absent nonsense and the
    present remainder -- because dropping the section entirely would also pass the first.
    """
    profile = QueryProfile(
        ops=(_op(0, "filter", 0, elapsed_ms=113.0, threads=64),),
        total_ms=37.0,
        rows=1,
        measured=True,
    )
    text = render_profile(profile, analyze=True)
    assert "where the time went" in text
    shares = [int(m) for m in re.findall(r"(\d+)%", text)]
    assert shares, "positive control: the section must still print a share at all"
    assert max(shares) <= 100, f"share above 100% in: {text}"
    # 113ms over 64 workers occupies ~1.8ms of a 37ms clock, so the remainder is real.
    assert "elsewhere" in text
    assert "planning, optimization, admission, FFI crossing, result assembly" in text


def test_a_sequential_profile_renders_exactly_as_it_did_before_the_wall_conversion():
    """`threads <= 1` makes the occupancy conversion the identity — the control for it."""
    profile = QueryProfile(
        ops=(_op(0, "scan", 0, elapsed_ms=40.0, threads=1),), total_ms=100.0, rows=10, measured=True
    )
    text = render_profile(profile, analyze=True)
    assert "40" in text and "60%" in text  # 40ms in operators, 60ms elsewhere


def test_a_profile_whose_operators_cover_the_clock_reports_no_remainder():
    profile = QueryProfile(
        ops=(_op(0, "scan", 0, elapsed_ms=100.0),), total_ms=100.0, rows=10, measured=True
    )
    text = render_profile(profile, analyze=True)
    assert "operators" in text and "elsewhere" not in text


# --- large plans -------------------------------------------------------------


def _wide_plan(cold: int = 30) -> QueryProfile:
    """One hot chain plus `cold` sibling scans that together are under 1% of the run."""
    ops = [
        _op(0, "aggregate", 0, elapsed_ms=1000.0, est_rows=10.0),
        _op(1, "union", 1, elapsed_ms=1.0),
        _op(2, "scan", 2, elapsed_ms=500.0, est_rows=10.0),
    ]
    ops += [_op(3 + i, "scan", 2, elapsed_ms=0.01, est_rows=10.0) for i in range(cold)]
    return QueryProfile(ops=tuple(ops), total_ms=1600.0, rows=10, measured=True)


def test_a_large_plan_folds_its_cold_subtrees_into_one_marker():
    text = render_profile(_wide_plan(), analyze=True, width=120)
    assert text.count("… ") == 1, "a run of cold siblings collapses to one line, not one each"
    assert "30 more" in text
    assert "30 operators folded" in text
    assert 'explain(format="json") lists every one' in text


def test_folding_never_hides_the_hot_path_or_a_root():
    text = render_profile(_wide_plan(), analyze=True, width=120)
    for kept in ("aggregate", "union"):
        assert kept in text
    hot = next(line for line in text.split("\n") if "scan" in line and "500ms" in line)
    assert hot, "the expensive scan must survive the fold"


def test_folding_is_off_for_a_small_plan_and_when_asked_to_be():
    small = render_profile(_join_plan(), analyze=True, width=120)
    assert "… " not in small, "a plan a reader can take in at once is never elided"
    full = render_profile(_wide_plan(), analyze=True, width=120, fold=False)
    assert "… " not in full
    assert full.count("scan") >= 31


def test_folding_is_off_without_measurements_because_coldness_is_unknowable():
    planned = QueryProfile(ops=_wide_plan().ops, total_ms=0.0, measured=False)
    assert "… " not in render_profile(planned, analyze=False, width=120)


def test_a_large_plan_names_its_hot_operators_before_the_tree():
    text = render_profile(_wide_plan(), analyze=True, width=120)
    assert "hot operators" in text
    _, tree = text.split("hot operators", 1)
    assert "aggregate (op 0)" in tree.split("\n\n")[0]


def test_a_small_plan_does_not_repeat_itself_as_a_hot_operator_table():
    assert "hot operators" not in render_profile(_join_plan(), analyze=True, width=120)


# --- diagnosis ---------------------------------------------------------------


def test_a_badly_missed_estimate_is_called_out_with_its_direction():
    text = render_profile(_join_plan(), analyze=True, width=140)
    assert "what to look at" in text
    assert "5000.0x over" in text
    assert "hash_join (op 2) row estimate was" in text


def test_a_close_estimate_raises_nothing():
    ops = (_op(0, "scan", 0, est_rows=10.0, rows_out=11),)
    text = render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True), analyze=True)
    assert "what to look at" not in text


def test_a_spill_says_how_much_and_what_to_do():
    ops = (_op(0, "aggregate", 0, est_rows=10.0, spilled=True, spill_bytes=2 * 1024**3),)
    text = render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True), analyze=True)
    assert "spill 2.0 GiB" in text
    assert "raise the memory envelope" in text


def test_paging_is_reported_as_critical_because_it_invalidates_the_timings():
    ops = (_op(0, "aggregate", 0, est_rows=10.0, major_faults=4096, threads=4),)
    text = render_profile(QueryProfile(ops=ops, total_ms=1.0, measured=True), analyze=True)
    assert "PAGING" in text
    assert "the machine was paging against this query" in text


# --- presentation seam -------------------------------------------------------


def test_styling_is_injected_and_defaults_to_adding_nothing():
    """`plan` may not import `observe`, so color arrives as a callable or not at all."""
    plain = render_profile(_join_plan(), analyze=True, width=120)
    assert "\x1b" not in plain

    def loud(role: str, text: str) -> str:
        return f"<{role}>{text}</{role}>" if text else text

    styled = render_profile(_join_plan(), analyze=True, width=120, style=loud)
    assert "<head>" in styled and "<muted>" in styled


def test_the_rule_underlines_the_table_not_the_terminal():
    """A width-dependent rule makes the same plan render differently in every window."""
    narrow = render_profile(_join_plan(), analyze=False, width=80)
    wide = render_profile(_join_plan(), analyze=False, width=160)
    assert narrow.split("\n")[1] == wide.split("\n")[1]


def test_planned_output_never_claims_a_column_it_does_not_have():
    """A header for a column that does not exist is how a reader believes a plan was run."""
    planned = render_profile(_join_plan(), analyze=False)
    assert "actual" not in planned and "ACTUAL" not in planned
    assert "OP SHARE" not in planned


def test_an_empty_profile_renders_rather_than_raising():
    text = render_profile(QueryProfile(ops=()), analyze=False)
    assert "(no operators)" in text


def test_render_options_expose_the_glyph_table_they_selected():
    assert RenderOptions(unicode=False).glyphs["tee"] == "|- "
    assert RenderOptions(unicode=True).glyphs["tee"] == "├─ "
