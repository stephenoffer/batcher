"""The guards and errors around `join`'s keywords, `join_where` and `update(validate=)`.

The results themselves are checked against DuckDB in
`tests/differential/test_diff_join_keywords.py`; this file pins what is refused, and how.
"""

from __future__ import annotations

import random

import pytest

import batcher as bt
from batcher._internal.errors import DataQualityError, PlanError


def _sides() -> tuple[bt.Dataset, bt.Dataset]:
    left = bt.from_pydict({"k": [1, 2, 2, None, None], "a": [1, 2, 3, 4, 5]})
    right = bt.from_pydict({"k": [1, 3, None, None], "b": [10, 30, 40, 50]})
    return left, right


# --- join_where: a name-only reference compared with itself (AP-092) ---


def test_join_where_refuses_a_same_name_self_comparison():
    left = bt.from_pydict({"k": [1, 2]})
    right = bt.from_pydict({"k": [1, 3]})
    with pytest.raises(PlanError, match=r"bt\.col\('k_right'\)"):
        left.join_where(right, left["k"] == right["k"])
    with pytest.raises(PlanError, match="LEFT column 'k'"):
        left.join_where(right, bt.col("k") < bt.col("k"))


def test_join_where_suggests_the_callers_suffix():
    left = bt.from_pydict({"k": [1]})
    right = bt.from_pydict({"k": [1]})
    with pytest.raises(PlanError, match=r"bt\.col\('k_r'\)"):
        left.join_where(right, bt.col("k") == bt.col("k"), suffix="_r")


def test_join_where_suffixed_reference_still_joins():
    left = bt.from_pydict({"k": [1, 2]})
    right = bt.from_pydict({"k": [1, 3]})
    out = left.join_where(right, bt.col("k") == bt.col("k_right")).to_pydict()
    assert out == {"k": [1], "k_right": [1]}


def test_join_where_self_comparison_of_a_one_sided_column_is_not_refused():
    """Only a name both sides have is ambiguous; a left-only column compared to itself is not."""
    left = bt.from_pydict({"k": [1, 2], "x": [5, 6]})
    right = bt.from_pydict({"k": [1, 3]})
    out = left.join_where(right, bt.col("x") == bt.col("x"), bt.col("k") == bt.col("k_right"))
    assert out.count() == 1


# --- validate= (AP-101) ---


@pytest.mark.parametrize(
    ("mode", "side"), [("1:1", "left"), ("1:m", "left"), ("m:1", "right"), ("1:1", "right")]
)
def test_validate_names_the_side_and_the_repeated_key(mode, side):
    left, right = _sides()
    right = right.union(bt.from_pydict({"k": [3], "b": [31]}))
    if side == "right":
        left = left.distinct(["k"], keep="first", order_by="a")
    with pytest.raises(DataQualityError, match=f"the {side} side must have unique keys"):
        left.join(right, on="k", validate=mode)


def test_validate_quotes_the_repeated_keys():
    left, right = _sides()
    with pytest.raises(DataQualityError, match=r"e\.g\. \[2\]"):
        left.join(right, on="k", validate="1:m")


def test_validate_ignores_repeated_null_keys():
    """Two null keys on the right match nothing, so they cannot expand a row."""
    left, right = _sides()
    assert left.join(right, on="k", how="left", validate="m:1").count() == 5


def test_validate_counts_null_keys_when_nulls_are_equal():
    left, right = _sides()
    with pytest.raises(DataQualityError, match=r"right side.*\[None\]"):
        left.join(right, on="k", validate="m:1", nulls_equal=True)


def test_validate_default_checks_nothing():
    left, right = _sides()
    assert left.join(right, on="k").count() == 1


def test_validate_rejects_an_unknown_mode():
    left, right = _sides()
    with pytest.raises(PlanError, match="validate must be one of"):
        left.join(right, on="k", validate="one_to_one")


def test_validate_names_an_expression_key_as_written():
    left = bt.from_pydict({"s": ["A", "a"]})
    right = bt.from_pydict({"s": ["a"]})
    key = bt.col("s").str.lower()
    with pytest.raises(DataQualityError, match="left side") as info:
        left.join(right, left_on=key, right_on=key, validate="1:1")
    assert "__bc_jkey" not in str(info.value)


def test_validate_on_a_multi_key_join():
    left = bt.from_pydict({"a": [1, 1, 1], "b": [1, 2, 2]})
    right = bt.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(DataQualityError, match=r"\[\(1, 2\)\]"):
        left.join(right, on=["a", "b"], validate="1:1")
    assert left.join(right, on=["a", "b"], validate="m:1").count() == 2


# --- update(validate=) (AP-108) ---


def test_update_validate_reports_duplicate_source_keys():
    prices = bt.from_pydict({"id": [1, 2], "price": [10, 20]})
    fixes = bt.from_pydict({"id": [2, 2], "price": [25, 26]})
    # The documented default still repeats the row ...
    assert prices.update(fixes, on="id").count() == 3
    # ... and validate= refuses it, quoting the key.
    with pytest.raises(DataQualityError, match=r"right side.*\[2\]"):
        prices.update(fixes, on="id", validate="m:1")


# --- expression keys (AP-104) ---


def test_bare_col_key_is_the_name():
    left, right = _sides()
    by_name = left.join(right, on="k").to_pydict()
    assert left.join(right, on=bt.col("k")).to_pydict() == by_name
    assert left.join(right, left_on=[bt.col("k")], right_on="k").to_pydict() == by_name


def test_verbs_without_expression_keys_say_so():
    left = bt.from_pydict({"k": [1], "t": [1]})
    right = bt.from_pydict({"k": [1], "t": [1]})
    with pytest.raises(PlanError, match="takes column names as keys"):
        left.update(right, left_on=bt.col("k") + 1, right_on="k")


def test_a_non_key_object_is_refused_clearly():
    left, right = _sides()
    with pytest.raises(PlanError, match="column names or expressions"):
        left.join(right, on=[1])


# --- keywords a join type cannot honour ---


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_semi_anti_refuse_indicator_and_coalesce_false(how):
    left, right = _sides()
    with pytest.raises(PlanError, match="indicator"):
        left.join(right, on="k", how=how, indicator="src")
    with pytest.raises(PlanError, match="coalesce=False"):
        left.join(right, on="k", how=how, coalesce=False)


def test_cross_refuses_key_keywords():
    left, right = _sides()
    with pytest.raises(PlanError, match="cross"):
        left.join(right, how="cross", indicator="src")
    with pytest.raises(PlanError, match="cross"):
        left.join(right, how="cross", validate="1:1")


def test_indicator_name_must_be_free():
    left, right = _sides()
    with pytest.raises(PlanError, match="indicator='a'"):
        left.join(right, on="k", how="full", indicator="a")


def test_nulls_equal_refuses_a_nested_key():
    left = bt.from_pydict({"k": [[1], None]})
    right = bt.from_pydict({"k": [[1], None]})
    with pytest.raises(PlanError, match="cannot compare key 'k'"):
        left.join(right, on="k", nulls_equal=True)


def test_nulls_equal_list_must_match_the_keys():
    left, right = _sides()
    with pytest.raises(PlanError, match="nulls_equal list has 2 entries but there are 1 keys"):
        left.join(right, on="k", nulls_equal=[True, False])


def test_coalesce_true_is_the_default():
    left, right = _sides()
    for how in ("inner", "left", "right", "full"):
        default = left.join(right, on="k", how=how)
        forced = left.join(right, on="k", how=how, coalesce=True)
        assert forced.columns == default.columns
        assert sorted(map(str, forced.to_pylist())) == sorted(map(str, default.to_pylist()))


# --- join_asof ties (AP-113) ---


def test_asof_tie_recipe_is_independent_of_input_order():
    """Deduplicating ties by a sequence number picks the same row however rows arrive."""
    rows = [(4, 1, "old"), (4, 2, "new"), (4, 3, "newest"), (8, 1, "x")]
    left = bt.from_pydict({"t": [5, 9]})
    picks = set()
    for seed in range(4):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        quotes = bt.from_pydict(
            {
                "t": [r[0] for r in shuffled],
                "seq": [r[1] for r in shuffled],
                "w": [r[2] for r in shuffled],
            }
        )
        latest = quotes.distinct(["t"], keep="last", order_by="seq")
        picks.add(tuple(left.join_asof(latest, on="t").sort("t").to_pydict()["w"]))
    assert picks == {("newest", "x")}


def test_full_join_suffixes_a_right_column_named_like_the_left_key():
    """Regression: the coalesced key's name was not reserved, so this raised a duplicate."""
    left = bt.from_pydict({"k": [1, 2], "v": [1, 2]})
    right = bt.from_pydict({"rk": [2, 3], "k": [9, 8]})
    out = left.join(right, left_on="k", right_on="rk", how="full")
    assert out.columns == ["k", "v", "k_right"]
    rows = sorted(out.to_pylist(), key=lambda r: r["k"])
    assert rows == [
        {"k": 1, "v": 1, "k_right": None},
        {"k": 2, "v": 2, "k_right": 9},
        {"k": 3, "v": None, "k_right": 8},
    ]
