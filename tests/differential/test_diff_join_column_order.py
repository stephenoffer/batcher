"""Where a `USING` join puts its key column, and why the oracle does not decide it.

`assert_same` compares column names as a **set**, so nothing in the differential suite
checks where a join's output columns land. That is deliberate, and this file is the record
of why — plus the assertions that keep *Batcher's* answer pinned now that the oracle has
stopped watching it.

Tightening `assert_same` to compare column names positionally was tried. Every explicit
select list in the suite passed unchanged (2,334 SQL tests), and **49 join tests failed**,
all on one difference:

    SELECT * FROM emp JOIN dept USING (dept_id)
    duckdb  -> ['id', 'dept_id', 'dept']    # the key stays in its left-table position
    batcher -> ['dept_id', 'id', 'dept']    # the coalesced key comes first

**Batcher is the one following the specification.** SQL:2016 §7.7 says a `USING` join's
coalesced columns come first, then the left table's remaining columns, then the right's;
PostgreSQL does the same. DuckDB does not, and it is entitled not to — but that makes its
column order the wrong thing to hold ourselves against. A positional oracle here would have
recorded 49 failures against Batcher for being correct, which is the failure mode the
harness already has a scar from: on TPC-H q6 a comparator's wrong answer was reported as
Batcher's until the oracle was pinned down.

So the property is asserted directly rather than differentially. The divergence is stated
as a fact about both engines, so if *either* changes its mind this file says so — and a
change to Batcher's own ordering, which no other test would now catch, fails here.

Measured against duckdb 1.5.5.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

_EMP = pa.table({"id": [1, 2, 3], "dept_id": [10, 20, 10], "salary": [100, 200, 150]})
_DEPT = pa.table({"dept_id": [10, 20], "dept": ["eng", "ops"]})


@pytest.fixture
def registered(duck):
    duck.register("emp", _EMP)
    duck.register("dept", _DEPT)
    return duck


def test_a_using_join_puts_the_coalesced_key_first(registered):
    """SQL:2016 §7.7, and what PostgreSQL does. This is the contract, not DuckDB's answer."""
    for got in (
        bt.from_arrow(_EMP).join(bt.from_arrow(_DEPT), on="dept_id").collect(),
        bt.sql("SELECT * FROM emp JOIN dept USING (dept_id)", emp=_EMP, dept=_DEPT).collect(),
    ):
        assert got.column_names == ["dept_id", "id", "salary", "dept"], (
            "the coalesced key comes first, then the left table's remainder, then the right's"
        )


def test_duckdb_orders_it_differently_and_that_is_not_our_defect(registered):
    """The other half of the divergence, so the comparison stays honest in both directions.

    If DuckDB ever adopts the standard ordering this test fails, and the right response is
    to delete it and let `assert_same` go positional — which would then be free, and would
    close the permuted-projection hole that the set comparison leaves open.

    **`benchmarks/harness/divergences.py` carries the same divergence** as an entry citing
    SQL:2016 §7.7 rule 1.b with verdict "batcher", so the benchmark suite reports a TPC-H row
    hitting it as `DIVERGENT` rather than `FAILED`. The two agree by construction and have to
    be retired together: deleting this test without that entry would leave the benchmark
    excusing a difference nothing documents any more.
    """
    theirs = registered.sql("SELECT * FROM emp JOIN dept USING (dept_id)").to_arrow_table()
    assert theirs.column_names == ["id", "dept_id", "salary", "dept"], (
        "DuckDB keeps the join key at its original left-table index — it does not move it "
        "to the front as the standard says, nor to the back; if this changed, revisit "
        "`_harness.assert_same`"
    )


def test_the_two_engines_still_agree_on_the_rows(registered):
    """The divergence is presentational only: same rows, same values, different order.

    Worth asserting explicitly, because "the column order differs" would be a much more
    serious finding if it turned out to mean the *columns* differed.
    """
    mine = bt.from_arrow(_EMP).join(bt.from_arrow(_DEPT), on="dept_id").collect().to_pydict()
    theirs = (
        registered.sql("SELECT * FROM emp JOIN dept USING (dept_id)").to_arrow_table().to_pydict()
    )
    assert set(mine) == set(theirs)
    for name in mine:
        assert sorted(mine[name], key=repr) == sorted(theirs[name], key=repr), name


def test_an_explicit_select_list_is_honoured_in_the_order_written(registered):
    """The case that *is* a promise, and the one `assert_same_for_query` now checks.

    A star leaves the order to the engine; a named list does not, and both engines agree
    here — which is what made the failed tightening attempt so informative: the 49 failures
    were entirely stars.
    """
    query = "SELECT dept, id, dept_id FROM emp JOIN dept USING (dept_id)"
    mine = bt.sql(query, emp=_EMP, dept=_DEPT).collect()
    theirs = registered.sql(query).to_arrow_table()
    assert mine.column_names == theirs.column_names == ["dept", "id", "dept_id"]
