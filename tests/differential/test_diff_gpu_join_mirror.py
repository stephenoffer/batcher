"""A mirrored GPU join computes the same rows as the plan-ordered one, and as DuckDB.

`tests/unit/test_gpu_join_mirror.py` proves the IR transform is a faithful relabelling. That is
necessary and not sufficient: it checks the *description* of the join, and both descriptions
could be executed wrongly the same way. This runs the translator's actual join executor over
both forms and compares each against the oracle, which is what makes exchanging the sides a
scheduling decision rather than a semantic one.

The join runs on the translator's **host** backend, as the other mergeable device tests do. What
is under test is which relation gets replicated, a driver-side choice that is host code on every
run; a device would change which kernel merged the frames and not which rows come out.

Nulls in the key are included deliberately. The executor has a separate path for them — a
synthetic key component that stops a null matching itself — and mirroring exchanges which side
that marker is built on, so a null-bearing key is exactly where an asymmetry would hide.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same, assert_tables_equal
from batcher.core.gpu_plan import DfBackend
from batcher.core.gpu_plan.execute import _equi_join
from batcher.dist.gpu.join import _mirrored_join_ir

pytestmark = pytest.mark.differential


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


_JOIN_IR = {
    "op": "hash_join",
    "join_type": "inner",
    "left_keys": ["lk"],
    "right_keys": ["rk"],
    "output": [
        {"side": "left", "name": "lk", "alias": "k"},
        {"side": "right", "name": "label", "alias": "label"},
        {"side": "left", "name": "amount", "alias": "amount"},
    ],
}

_CASES = {
    "plain": (
        {"lk": [1, 2, 3, 4, 2], "amount": [10, 20, 30, 40, 50]},
        {"rk": [2, 3, 5], "label": ["b", "c", "e"]},
    ),
    "null keys on both sides": (
        {"lk": [1, None, 3, None], "amount": [10, 20, 30, 40]},
        {"rk": [None, 3, 1], "label": ["x", "y", "z"]},
    ),
    "duplicates on both sides": (
        {"lk": [1, 1, 2, 2], "amount": [1, 2, 3, 4]},
        {"rk": [1, 1, 2], "label": ["p", "q", "r"]},
    ),
    "no matches at all": (
        {"lk": [1, 2], "amount": [1, 2]},
        {"rk": [7, 8], "label": ["s", "t"]},
    ),
    "empty build side": (
        {"lk": [1, 2], "amount": [1, 2]},
        {"rk": [], "label": []},
    ),
    "empty probe side": (
        {"lk": [], "amount": []},
        {"rk": [1], "label": ["u"]},
    ),
}


def _run(be, left: dict, right: dict, join_ir: dict) -> pa.Table:
    lf = be.from_arrow(pa.table(left))
    rf = be.from_arrow(pa.table(right))
    return be.to_arrow(_equi_join(lf, rf, join_ir, join_ir["join_type"], be))


def _duckdb_relation(left: dict, right: dict):
    """The oracle, as a DuckDB relation — which is what `assert_same` compares against."""
    import duckdb

    con = duckdb.connect()
    con.register("l", pa.table(left))
    con.register("r", pa.table(right))
    return con.sql(
        "SELECT l.lk AS k, r.label AS label, l.amount AS amount FROM l JOIN r ON l.lk = r.rk"
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_the_mirrored_join_matches_the_plan_ordered_one(be, case):
    """Exchanging the sides changes nothing about the rows."""
    left, right = _CASES[case]

    straight = _run(be, left, right, _JOIN_IR)
    mirrored = _run(be, right, left, _mirrored_join_ir(_JOIN_IR))

    assert mirrored.column_names == straight.column_names
    assert mirrored.schema.types == straight.schema.types
    assert_tables_equal(mirrored, straight)


@pytest.mark.parametrize("case", list(_CASES))
def test_both_orders_match_duckdb(be, case):
    """And the thing they agree on is the right answer.

    Without this the test above would be satisfied by two identically wrong joins, which is the
    failure mode a self-comparison cannot see.
    """
    left, right = _CASES[case]

    assert_same(_run(be, left, right, _JOIN_IR), _duckdb_relation(left, right))
    assert_same(_run(be, right, left, _mirrored_join_ir(_JOIN_IR)), _duckdb_relation(left, right))
