"""`ds.count()` asks the source for a `COUNT(*)` instead of reading it to count.

`Source.row_count` is answered while *planning*, and a warehouse source returns `None`
there deliberately: Kyber wants a free estimate and has a better one in the learned
statistics, so a query per plan would be a bad trade. But `ds.count()` is not an estimate,
and when the free metadata answers decline the fallback wraps the plan in a `COUNT(*)`
aggregate and *executes* it. A `COUNT(*)` needs no columns, so the projection that would
have narrowed the read is empty — which the SQL builder renders as ``SELECT *``. Counting
a warehouse table therefore pulled every column of every row across the network to return
one integer.

The gate is about plan shape. A projection cannot change how many rows there are, so a
scan under projections is answerable; a filter, aggregate, join or limit is not, and those
must decline rather than return the source's own count as though it were the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.metadata_answer import pushed_count
from batcher.io.formats.sql._common import COUNT_COLUMN, count_query

pytestmark = pytest.mark.unit


@dataclass
class _CountableSource:
    """A source that knows its own count and records whether it was read."""

    rows: int = 7
    supports_count: bool = True
    reads: list = field(default_factory=list)

    def exact_row_count(self) -> int | None:
        return self.rows

    def read(self, projection=None):
        self.reads.append(projection)
        return [pa.RecordBatch.from_pydict({"a": [1] * self.rows})]


def _plan_of(ds):
    return ds._plan


def test_a_bare_scan_is_answered_by_the_source():
    source = _CountableSource()
    plan = _plan_of(bt.from_pydict({"a": [1, 2, 3]}))
    assert pushed_count(plan, [source]) == 7
    assert source.reads == []  # nothing was read to produce it


def test_projections_do_not_change_the_count_so_they_pass_through():
    source = _CountableSource()
    plan = _plan_of(bt.from_pydict({"a": [1], "b": [2]}).select("a"))
    assert pushed_count(plan, [source]) == 7


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.filter(bt.col("a") > 1), id="filter"),
        pytest.param(lambda ds: ds.limit(2), id="limit"),
        pytest.param(lambda ds: ds.distinct(), id="distinct"),
        pytest.param(lambda ds: ds.group_by("a").agg(n=bt.col("a").count()), id="aggregate"),
    ],
)
def test_a_shape_that_changes_the_row_count_declines(build):
    source = _CountableSource()
    assert pushed_count(_plan_of(build(bt.from_pydict({"a": [1, 2, 3]}))), [source]) is None


def test_a_source_that_cannot_count_itself_declines():
    source = _CountableSource(supports_count=False)
    assert pushed_count(_plan_of(bt.from_pydict({"a": [1]})), [source]) is None


def test_a_source_whose_count_raises_falls_back_to_running_the_plan():
    class _Broken(_CountableSource):
        def exact_row_count(self):
            raise RuntimeError("server said no")

    # Counting is an optimization; a failure must degrade to the ordinary path.
    assert pushed_count(_plan_of(bt.from_pydict({"a": [1]})), [_Broken()]) is None


def test_the_count_query_selects_no_columns_and_names_its_result():
    sql = count_query("SELECT * FROM orders")
    assert sql.startswith(f"SELECT COUNT(*) AS {COUNT_COLUMN}")
    assert "SELECT * FROM orders" in sql


def test_the_count_alias_is_a_portable_identifier():
    # Oracle rejects an unquoted identifier starting with an underscore, so an alias
    # spelled `_bc_n` is a syntax error there rather than a portable name.
    assert not COUNT_COLUMN.startswith("_")
    assert COUNT_COLUMN.isidentifier()


def test_the_count_query_wraps_a_plain_table_read():
    assert "orders" in count_query(None, table="orders")


def test_a_sql_source_reads_its_count_off_the_one_row_the_server_returns():
    from batcher.io.formats.sql._source_base import SingleResultQuerySource

    seen: list[str] = []

    class _FakeSplit:
        def __init__(self, sql):
            self.sql = sql

        def read(self, projection=None):
            seen.append((self.sql, projection))
            # The server folds the alias' case; the reader must not look it up by name.
            return [pa.RecordBatch.from_pydict({COUNT_COLUMN.upper(): [1234]})]

    @dataclass(frozen=True, slots=True)
    class _FakeSource(SingleResultQuerySource):
        query: str = "SELECT * FROM t"

        def _split_for(self, sql: str):
            return _FakeSplit(sql)

        def identity(self) -> str:
            return "fake"

    assert _FakeSource().exact_row_count() == 1234
    assert seen[0][0].startswith("SELECT COUNT(*)")
    assert seen[0][1] is None  # read positionally, not by the folded alias


def test_the_sql_base_still_refuses_to_count_at_plan_time():
    from batcher.io.formats.sql._source_base import SingleResultQuerySource

    @dataclass(frozen=True, slots=True)
    class _FakeSource(SingleResultQuerySource):
        query: str = "SELECT * FROM t"

        def _split_for(self, sql: str):
            raise AssertionError("planning must not query the server")

        def identity(self) -> str:
            return "fake"

    # `row_count` is the plan-time question and stays free; `exact_row_count` is the
    # terminal one and is the only path allowed to spend a round trip.
    assert _FakeSource().row_count() is None
