"""What `ds.dq.suggest()` reads off a table, and what it refuses to guess.

A suggestion is only useful if it is true of the data, and only *safe* if it stops short of
the bounds that are true today by coincidence. So these tests pin both halves: everything
proposed holds when validated against the same relation, and the proposals that would age
badly — a range read off an observed minimum and maximum — are not made at all.
"""

from __future__ import annotations

import batcher as bt


def _trades():
    return bt.from_pydict(
        {
            "trade_id": [1, 2, 3, 4, 5, 6],
            "side": ["buy", "sell", "buy", "sell", "buy", "buy"],
            "price": [10.5, 11.0, 9.75, 12.25, 10.0, 11.5],
            "venue": ["LSE", None, "NYSE", None, "LSE", "LSE"],
            "notes": ["a", "b", "c", "d", "e", "f"],
        }
    )


def test_every_suggestion_holds_on_the_data_it_was_read_from():
    ds = _trades()
    proposed = ds.dq.suggest()
    report = proposed.validate()
    assert report.ok, report.violations
    assert report.total_violations == 0


def test_a_complete_column_gets_a_completeness_constraint():
    names = _trades().dq.suggest().validate().violations
    assert "not_null(trade_id)" in names
    assert "not_null(side)" in names


def test_a_sparse_column_gets_a_tolerated_null_rate_instead():
    names = _trades().dq.suggest().validate().violations
    assert "not_null(venue)" not in names
    # A third are null, so the bound is rounded up above it with headroom.
    rate = next(n for n in names if n.startswith("null_rate_below(venue"))
    assert float(rate.split(", ")[1].rstrip(")")) > 1 / 3


def test_a_key_is_proposed_as_unique():
    assert "unique(trade_id)" in _trades().dq.suggest().validate().violations


def test_a_small_vocabulary_is_proposed_as_an_enumeration():
    names = _trades().dq.suggest().validate().violations
    assert "accepted_values(side)" in names
    # `notes` has one distinct value per row, so it is free text, not an enum.
    assert "accepted_values(notes)" not in names


def test_no_range_is_read_off_an_observed_minimum_and_maximum():
    """Tomorrow's legitimate value is outside today's, and a check that cries wolf is deleted."""
    names = _trades().dq.suggest().validate().violations
    assert not any(n.startswith("in_range(") for n in names)


def test_a_signed_column_gets_no_sign_constraint():
    ds = bt.from_pydict({"pnl": [1.0, -2.0, 3.0]})
    names = ds.dq.suggest().validate().violations
    assert not any(n.startswith("positive(") or n.startswith("non_negative(") for n in names)


def test_a_float_column_is_asked_to_stay_finite():
    assert "is_finite(price)" in _trades().dq.suggest().validate().violations


def test_columns_can_be_restricted():
    names = _trades().dq.suggest(["side"]).validate().violations
    assert all("side" in n for n in names), names


def test_suggestions_extend_an_existing_chain():
    ds = _trades()
    chain = ds.dq.row_count_between(1).suggest(["trade_id"])
    names = list(chain.validate().violations)
    assert names[0] == "row_count_between(1, None)"
    assert "unique(trade_id)" in names


def test_an_empty_relation_proposes_nothing_it_cannot_support():
    empty = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 100)
    assert empty.dq.suggest().validate().ok
