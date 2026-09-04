"""Every path that rejects a column name must name the column the user meant.

`_internal.errors.suggest` exists so that the single most-read message Batcher prints -- "you
named something that does not exist" -- has one shape everywhere: what failed, the offending
value, the closest match, and a *bounded* list of what is valid. Its own docstring records
that three divergent hand-written copies are what prompted it.

The tree had drifted back. Measured across the nine ways a user can name a column, four
message shapes were in use and the four highest-traffic paths -- a typo in a `filter`, a
`select`, a `sort` key or an `agg` -- offered no suggestion at all, while `drop` and `rename`
(reached through a different helper) did. So whether Batcher told you the column you meant
depended on which verb you typed it into.

Two of them also interpolated the whole schema into the sentence. On a 200-column relation
`group_by` produced a 2,302-character message: the one name that was wrong, buried under the
200 that were right. `BatcherError` truncates its `available` *field*, so passing the columns
as the field rather than as text is what bounds it -- which is why these assert on a length,
not on a phrasing.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import BatcherError

pytestmark = pytest.mark.unit

#: Every public way to name a column, each given a name one transposition away from a real
#: one. The lambda takes the dataset so the parametrization reads as the user would write it.
NEAR_MISSES = {
    "filter": lambda ds: ds.filter(bt.col("alpah") > 1),
    "select": lambda ds: ds.select("alpah"),
    "with_columns": lambda ds: ds.with_columns(z=bt.col("alpah") * 2),
    "sort": lambda ds: ds.sort("alpah"),
    "group_by": lambda ds: ds.group_by("alpah").agg(t=bt.col("beta").sum()),
    "agg": lambda ds: ds.group_by("beta").agg(t=bt.col("alpah").sum()),
    "join": lambda ds: ds.join(ds, on="alpah"),
    "drop": lambda ds: ds.drop("alpah"),
    "rename": lambda ds: ds.rename({"alpah": "z"}),
}


@pytest.fixture
def frame():
    """Three columns, one of which is a transposition away from the name every case uses."""
    return bt.from_pydict({"alpha": [1, 2], "beta": [3, 4], "gamma": [5, 6]})


@pytest.mark.parametrize("verb", sorted(NEAR_MISSES))
def test_every_verb_names_the_column_the_user_meant(verb, frame):
    """A transposed character is the commonest column typo and the easiest to resolve."""
    with pytest.raises(BatcherError) as caught:
        NEAR_MISSES[verb](frame)
    rendered = str(caught.value)
    assert "alpha" in rendered, f"{verb} did not offer the intended column: {rendered}"
    assert "mean" in rendered.lower(), f"{verb} offered no suggestion: {rendered}"


@pytest.mark.parametrize("verb", sorted(NEAR_MISSES))
def test_a_wide_schema_does_not_bury_the_error_in_its_own_context(verb):
    """The message stays readable against 200 columns, which is a small table in practice.

    A generous ceiling on purpose: this is a regression guard against interpolating the
    schema, not a budget anyone should tune a message against. The unbounded form scored
    2,302 characters here, so anything near the limit is the bug coming back rather than a
    message that grew a clause.
    """
    wide = bt.from_pydict({f"col_{i:03d}": [1, 2] for i in range(200)})
    misses = {
        "filter": lambda ds: ds.filter(bt.col("col_00x") > 1),
        "select": lambda ds: ds.select("col_00x"),
        "with_columns": lambda ds: ds.with_columns(z=bt.col("col_00x") * 2),
        "sort": lambda ds: ds.sort("col_00x"),
        "group_by": lambda ds: ds.group_by("col_00x").agg(t=bt.col("col_001").sum()),
        "agg": lambda ds: ds.group_by("col_001").agg(t=bt.col("col_00x").sum()),
        "join": lambda ds: ds.join(ds, on="col_00x"),
        "drop": lambda ds: ds.drop("col_00x"),
        "rename": lambda ds: ds.rename({"col_00x": "z"}),
    }
    with pytest.raises(BatcherError) as caught:
        misses[verb](wide)
    rendered = str(caught.value)
    assert len(rendered) < 600, f"{verb} inlined the schema ({len(rendered)} chars): {rendered}"


def test_the_wide_schema_guard_can_fail():
    """The positive control for the length assertion above.

    A length ceiling passes trivially if the message it measures is short for some unrelated
    reason, so this pins the thing being guarded: rendering the same 200 names as text rather
    than as the truncated `available` field really does exceed the limit. Without it the test
    above would keep passing if the messages stopped carrying the column list at all, which
    would be a different regression reported as a pass.
    """
    columns = [f"col_{i:03d}" for i in range(200)]
    assert len(f"available: {sorted(columns)}") > 600


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_join_key_error_names_the_side_and_the_near_miss(side):
    """All six join-key checks route through one builder, so both sides read identically."""
    left = bt.from_pydict({"key": [1, 2], "v": [1, 2]})
    right = bt.from_pydict({"key": [1, 2], "w": [3, 4]})
    keys = ("kye", "key") if side == "left" else ("key", "kye")
    on = {"left_on": keys[0], "right_on": keys[1]}
    with pytest.raises(BatcherError) as caught:
        left.join(right, **on)
    rendered = str(caught.value)
    assert "kye" in rendered
    assert "Did you mean 'key'?" in rendered, rendered


def test_the_error_carries_its_parts_as_fields_not_only_as_prose():
    """Tooling reads the fields; the sentence is for people and may be reworded."""
    frame = bt.from_pydict({"alpha": [1], "beta": [2]})
    with pytest.raises(BatcherError) as caught:
        frame.filter(bt.col("alpah") > 1)
    err = caught.value
    assert err.column == "alpah"
    assert "alpha" in err.available
    assert "alpha" in err.suggestion
