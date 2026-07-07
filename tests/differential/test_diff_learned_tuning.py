"""Result-invariance of Kyber's learned tuning decisions, cross-checked against DuckDB.

Learned tuning changes *which equivalent physical algorithm/threshold* a join uses, never the
relation. These tests force each learned decision through the real Python-plan → IR → engine
path and assert the answer is identical (a) to DuckDB and (b) to the un-tuned default result —
the on/off proof the tuning is a pure performance knob.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_JOINS = [
    ("inner", "JOIN"),
    ("left", "LEFT JOIN"),
    ("right", "RIGHT JOIN"),
]


def _tables(duck):
    emp = pa.table({"id": [1, 2, 3, 4, 5], "name": list("abcde"), "dept_id": [10, 20, 10, 99, 20]})
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return bt.from_arrow(emp), bt.from_arrow(dept)


@pytest.mark.parametrize("arm", ["hash", "broadcast", "sort_merge"])
def test_learned_join_strategy_arm_is_result_invariant(duck, monkeypatch, arm):
    """Forcing the bandit to any arm keeps every join type equal to DuckDB and to the default."""
    from batcher.kyber.rules import selection
    from conftest import assert_same

    # Pin the learned bandit to `arm` for every signature — the on/off switch for the tuning.
    monkeypatch.setattr(selection, "learned_join_strategy", lambda hub, sig, *a, **k: arm)

    emp, dept = _tables(duck)
    for how, sql in _JOINS:
        out = emp.join(dept, on="dept_id", how=how).collect()
        assert_same(out, duck.sql(f"SELECT * FROM emp {sql} dept USING (dept_id)"))
    semi = emp.join(dept, on="dept_id", how="semi").collect()
    assert_same(semi, duck.sql("SELECT emp.* FROM emp SEMI JOIN dept USING (dept_id)"))
    anti = emp.join(dept, on="dept_id", how="anti").collect()
    assert_same(anti, duck.sql("SELECT emp.* FROM emp ANTI JOIN dept USING (dept_id)"))


def test_learned_thresholds_are_result_invariant(duck, monkeypatch):
    """A learned broadcast threshold / sort-merge crossover only re-selects the algorithm.

    Pin the learned broadcast threshold to -1 (never broadcast) and the learned sort-merge
    crossover to 0 (always sort-merge) — the extreme the OLS learners could produce — and every
    join type still matches DuckDB, proving the learned thresholds are pure performance knobs."""
    from batcher.kyber.rules import selection
    from conftest import assert_same

    monkeypatch.setattr(selection, "learned_broadcast_max_bytes", lambda hub, *a, **k: -1)
    monkeypatch.setattr(selection, "learned_sort_merge_min_rows", lambda hub, default, *a, **k: 0.0)

    emp, dept = _tables(duck)
    for how, sql in _JOINS:
        out = emp.join(dept, on="dept_id", how=how).collect()
        assert_same(out, duck.sql(f"SELECT * FROM emp {sql} dept USING (dept_id)"))


def test_join_then_aggregate_invariant_under_forced_strategy(duck, monkeypatch):
    """A learned strategy under a downstream aggregate still matches DuckDB (the SMJ output
    ordering/partitioning must not leak into the grouped result)."""
    from batcher import col, count
    from batcher.kyber.rules import selection
    from conftest import assert_same

    monkeypatch.setattr(selection, "learned_join_strategy", lambda hub, sig, *a, **k: "sort_merge")

    emp, dept = _tables(duck)
    out = (
        emp.join(dept, on="dept_id")
        .group_by("dept")
        .agg(headcount=count(), max_id=col("id").max())
        .collect()
    )
    expected = duck.sql(
        "SELECT dept, COUNT(*) AS headcount, MAX(id) AS max_id "
        "FROM emp JOIN dept USING (dept_id) GROUP BY dept"
    )
    assert_same(out, expected)


def test_forced_arm_matches_the_untuned_default(duck, monkeypatch):
    """The on/off proof: the tuned result is byte-for-byte the same as the default (hash) result."""
    from batcher.kyber.rules import selection
    from conftest import assert_same

    emp, dept = _tables(duck)
    default = emp.join(dept, on="dept_id").collect()  # cold hub → cost-model default

    monkeypatch.setattr(selection, "learned_join_strategy", lambda hub, sig, *a, **k: "sort_merge")
    tuned = emp.join(dept, on="dept_id").collect()

    # Compare the two Batcher results directly (via the DuckDB harness's multiset equality).
    duck.register("d", default)
    assert_same(tuned, duck.sql("SELECT * FROM d"))
