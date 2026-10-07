"""`register_function(null_handling=...)` against DuckDB's `create_function` of the same name.

DuckDB's default is NULL-in, NULL-out: a row with any NULL argument answers NULL and the
function never sees it. Batcher's default hands the NULL to the function (DuckDB's
``"special"``), and ``null_handling="default"`` opts into DuckDB's behaviour. Both engines
register the same Python callable with the same mode, so DuckDB is a real oracle here.

Every function records what it was handed, because the contract is also about *what reaches
`fn`*, not only what comes out: a ``"default"`` function must never see a NULL.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_CASES = {
    "nulls": {"x": [1, None, 3, None], "y": [None, 2, 3, None]},
    "empty": {"x": [], "y": []},
    "one_row": {"x": [5], "y": [None]},
    "duplicates": {"x": [2, 2, None, 2], "y": [1, 1, 1, None]},
}


def _table(case: str) -> pa.Table:
    return pa.table({k: pa.array(v, pa.int64()) for k, v in _CASES[case].items()})


def _row_add(seen: list):
    def add(a, b):
        seen.append((a, b))
        return (a or 0) + (b or 0)

    return add


def _vector_add(seen: list):
    def add(a, b):
        seen.append(a.null_count + b.null_count)
        return pc.add(pc.fill_null(a, 0), pc.fill_null(b, 0))

    return add


def _batcher(table: pa.Table, fn, *, vectorized: bool, mode: str, query: str):
    s = bt.Session()
    s.register("t", table)
    s.register_function("f", fn, vectorized=vectorized, result_type="int64", null_handling=mode)
    return s.sql(query).collect()


def _duckdb(duck, table: pa.Table, fn, *, vectorized: bool, mode: str, query: str):
    duck.register("t", table)
    kind = "arrow" if vectorized else "native"
    duck.create_function("f", fn, ["BIGINT", "BIGINT"], "BIGINT", type=kind, null_handling=mode)
    return duck.sql(query)


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("mode", ["default", "special"])
def test_null_handling_matches_duckdb(duck, case: str, vectorized: bool, mode: str) -> None:
    table = _table(case)
    query = "SELECT x, y, f(x, y) AS s FROM t"
    make = _vector_add if vectorized else _row_add
    seen_bt: list = []
    seen_duck: list = []
    ours = _batcher(table, make(seen_bt), vectorized=vectorized, mode=mode, query=query)
    theirs = _duckdb(duck, table, make(seen_duck), vectorized=vectorized, mode=mode, query=query)
    assert_same(ours, theirs)
    if mode == "default":
        # The point of the mode: the function is never handed a NULL.
        nulls_seen = seen_bt if vectorized else [p for p in seen_bt if None in p]
        assert not [n for n in nulls_seen if n]


def test_a_null_literal_argument_answers_null_without_a_call(duck) -> None:
    table = _table("nulls")
    query = "SELECT x, f(x, NULL) AS s FROM t"
    seen: list = []
    ours = _batcher(table, _row_add(seen), vectorized=False, mode="default", query=query)
    theirs = _duckdb(duck, table, _row_add([]), vectorized=False, mode="default", query=query)
    assert_same(ours, theirs)
    assert seen == []


def test_the_default_still_hands_nulls_to_the_function() -> None:
    # Batcher's default is DuckDB's "special": unchanged by the new keyword.
    seen: list = []
    s = bt.Session()
    s.register("t", _table("nulls"))
    s.register_function("f", _row_add(seen), vectorized=False, result_type="int64")
    s.sql("SELECT f(x, y) AS s FROM t").collect()
    assert (1, None) in seen


def test_null_handling_is_refused_where_it_cannot_apply() -> None:
    from batcher._internal.errors import PlanError

    s = bt.Session()
    with pytest.raises(PlanError, match="null_handling must be one of"):
        s.register_function("f", pc.add, null_handling="skip")
    with pytest.raises(PlanError, match="table function"):
        s.register_function("g", lambda b: b, table=True, null_handling="default")
