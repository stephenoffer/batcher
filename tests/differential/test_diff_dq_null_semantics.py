"""`ds.dq` against the null and empty cases its own docstrings make promises about.

`test_diff_dq` covers the happy path of each terminal. This covers the promises that are
*written down in the implementation* and were nobody's test — every one of them a place
where the wrong answer is a quiet one rather than an exception:

- a NULL is valid for a value constraint but a violation for a user `check()` predicate,
  because a comparison against NULL is NULL and a NULL validity is not a pass;
- two NULL keys are duplicates of each other for `unique`, since grouping treats them as
  one key;
- the uniqueness lowering adds `__dq_one` / `__dq_uniq_N` window helpers, which must not
  survive into either side of the result;
- the three terminals are three readings of one predicate, so `validate` must count
  exactly what `drop` removes and `quarantine` must partition without losing a row.

That last one carries the most weight. `validate`, `drop` and `quarantine` each consult
`_provably_clean`, a metadata shortcut that can answer "nothing violates this" from
footer statistics without reading a row. A shortcut that is wrong does not raise — it
returns clean data and an empty reject set, which is exactly what a passing contract
looks like. Asserting the three against each other on data that genuinely violates is
what makes such a divergence visible.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


def _n(ds) -> int:
    return ds.agg(n=bt.count()).to_pydict()["n"][0]


# ----------------------------------------------------------------- NULL is not a pass
def test_check_predicate_that_is_null_counts_as_a_violation() -> None:
    """`x > 0` is NULL where x is NULL, and a NULL validity must not be treated as TRUE."""
    ds = bt.from_pydict({"x": [1, None, 3]})
    assert ds.dq.check(bt.col("x") > 0, name="pos").drop().to_pydict() == {"x": [1, 3]}
    assert ds.dq.check(bt.col("x") > 0, name="pos").validate().violations == {"pos": 1}


def test_value_constraints_pass_a_null_but_not_null_catches_it() -> None:
    """The documented split: value constraints compose, `not_null` is the explicit gate."""
    ds = bt.from_pydict({"x": [1, None, 50]})
    assert ds.dq.in_range("x", 0, 10).drop().to_pydict() == {"x": [1, None]}
    assert ds.dq.in_range("x", 0, 10).not_null("x").drop().to_pydict() == {"x": [1]}


def test_two_null_keys_are_duplicates_of_each_other() -> None:
    """`unique` lowers to a count over the key partition, which groups NULLs together."""
    ds = bt.from_pydict({"k": [None, None, 1]})
    assert ds.dq.unique("k").drop().to_pydict() == {"k": [1]}
    assert ds.dq.unique("k").validate().violations == {"unique(k)": 1}


# -------------------------------------------------------------- the lowering's leftovers
def test_uniqueness_helper_columns_reach_neither_side_of_a_quarantine() -> None:
    """`__dq_one` and `__dq_uniq_N` are scaffolding; leaking them changes the user's schema."""
    ds = bt.from_pydict({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    clean, bad = ds.dq.unique("id").quarantine()
    for side in (clean, bad):
        assert [c for c in side.to_pydict() if c.startswith("__dq")] == []
        assert sorted(side.to_pydict()) == ["id", "v"]


def test_drop_preserves_the_input_column_order() -> None:
    """The window helpers are appended and dropped, so the surviving order must be untouched."""
    ds = bt.from_pydict({"b": [1, 2], "a": [3, 4], "c": [5, 6]})
    assert list(ds.dq.unique("a").in_range("b", 0, 9).drop().to_pydict()) == ["b", "a", "c"]


# --------------------------------------------------- the three terminals must agree
@pytest.mark.parametrize(
    "data",
    [
        {"x": [1, -1, 5, None, 20]},
        {"x": [None, None, None]},
        {"x": [1, 2, 3]},
        {"x": []},
    ],
    ids=["mixed", "all-null", "all-clean", "empty"],
)
def test_validate_drop_and_quarantine_are_three_readings_of_one_predicate(data) -> None:
    """Counted, removed, and set aside must be the same rows — including via the metadata
    shortcut, which can answer "clean" without reading anything and would not raise if wrong."""
    ds = bt.from_pydict({"x": pa.array(data["x"], pa.int64())})
    total = _n(ds)
    chain = ds.dq.in_range("x", 0, 10)

    kept = _n(chain.drop())
    clean, bad = chain.quarantine()
    n_clean, n_bad = _n(clean), _n(bad)
    violations = chain.validate().total_violations

    assert n_clean + n_bad == total  # a total partition: nothing lost or duplicated
    assert kept == n_clean  # drop and the clean side agree
    assert violations == total - kept  # and the count matches what was removed


def test_an_empty_dataset_satisfies_every_constraint() -> None:
    """Vacuous truth: no rows means no violating rows, so a contract gate must not fail."""
    ds = bt.from_pydict({"x": pa.array([], pa.int64())})
    report = ds.dq.not_null("x").unique("x").in_range("x", 0, 10).validate()
    assert report.ok and report.total_violations == 0
    assert _n(ds.dq.not_null("x").fail()) == 0


def test_a_constraint_naming_a_column_that_does_not_exist_is_an_error() -> None:
    """Silently passing a check written against a typo'd column is the worst outcome here."""
    ds = bt.from_pydict({"amount": [1]})
    with pytest.raises(PlanError) as exc:
        ds.dq.not_null("ammount").validate()
    # And it names the near-miss, so the typo is the obvious reading of the failure.
    assert "ammount" in str(exc.value) and "amount" in str(exc.value)
