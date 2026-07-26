"""Plan-shape tests for `prefix_predicates_to_range` and `len_zero_to_empty_string`.

Correctness lives in `tests/differential/test_diff_kyber_prefix_and_len_ranges.py`. What
these assert is the *reason* the rules exist: a bare `Col` on one side of a comparison,
which is the only shape zone-map pruning, bloom probing and source pushdown can use. A
predicate that returns the right rows through a function wrapper is exactly the failure
mode here — correct and unprunable.
"""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer

DS = bt.from_pydict({"s": ["ab", "abc", "", None], "n": [1, 2, 3, 4]})


def _plan(ds) -> str:
    return json.dumps(Optimizer().optimize(ds._plan).ir)


def _is_range(plan: str) -> bool:
    return '"op": "ge"' in plan and '"op": "lt"' in plan


def test_starts_with_becomes_the_same_range_as_like():
    """The DataFrame spelling now gets what the SQL spelling always got."""
    by_starts_with = _plan(DS.filter(col("s").str.starts_with("ab")))
    by_like = _plan(DS.filter(col("s").str.like("ab%")))
    assert _is_range(by_starts_with)
    assert by_starts_with == by_like, "the two spellings must optimize to one plan"


def test_a_first_character_substring_equality_becomes_a_range():
    plan = _plan(DS.filter(col("s").str.substr(1, 2) == "ab"))
    assert _is_range(plan)
    assert '"substr"' not in plan


def test_a_substring_not_anchored_at_the_first_character_is_left_alone():
    """`substr(s, 2, 2)` is not a prefix test, so no range may be derived from it."""
    plan = _plan(DS.filter(col("s").str.substr(2, 2) == "ab"))
    assert '"substr"' in plan
    assert not _is_range(plan)


def test_a_literal_of_the_wrong_length_is_left_alone():
    """`substr(s, 1, 2) = 'abc'` is unsatisfiable, not a prefix test.

    Turning it into a range for `'abc'` would *widen* it to strings starting `abc`, which
    is a different — and wrong — predicate.
    """
    plan = _plan(DS.filter(col("s").str.substr(1, 2) == "abc"))
    assert '"substr"' in plan


def test_a_prefix_whose_last_character_cannot_be_incremented_is_left_alone():
    """The upper bound is built by incrementing the last character, so the rule shares
    the `LIKE` rule's conservative ASCII guard rather than inventing its own."""
    plan = _plan(DS.filter(col("s").str.starts_with("aÿ")))
    assert '"starts_with"' in plan


def test_len_zero_unwraps_the_column():
    plan = _plan(DS.filter(col("s").str.len() == 0))
    assert '"fn": "len"' not in plan
    assert '{"str": ""}' in plan


def test_len_not_zero_unwraps_too():
    plan = _plan(DS.filter(col("s").str.len() != 0))
    assert '"fn": "len"' not in plan


def test_a_non_zero_length_comparison_is_left_alone():
    """`len(s) = 3` is not an equality on `s`; only the zero case has a string form."""
    plan = _plan(DS.filter(col("s").str.len() == 3))
    assert '"fn": "len"' in plan


def test_a_length_test_on_a_non_string_column_is_untouched():
    """Guard against matching a same-named function on another type."""
    plan = _plan(DS.filter(col("n") == 0))
    assert '"fn": "len"' not in plan  # nothing to unwrap; the rule must not invent one
    assert '"op": "eq"' in plan
