"""`bt.running_queries` and `bt.cancel_query`: the registry contract, without a long query.

`tests/integration/test_query_cancellation.py` covers cancelling a query mid-flight from
another thread. This file pins the registry half on its own, deterministically and in the
fast suite: an id is listed exactly while its scope is open, cancelling a listed id reports
True and makes the query raise rather than return short rows, and cancelling anything else
reports False instead of raising.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import QueryCancelledError
from batcher.core.runtime import query_scope

pytestmark = pytest.mark.unit


def test_nothing_is_listed_when_nothing_runs():
    assert bt.running_queries() == []


def test_an_open_scope_is_listed_and_released():
    with query_scope() as query_id:
        assert bt.running_queries() == [query_id]
    assert query_id not in bt.running_queries()


def test_a_cancelled_scope_raises_instead_of_returning_rows():
    ds = bt.from_pydict({"a": list(range(50_000))}).filter(bt.col("a") > 0)
    with pytest.raises(QueryCancelledError), query_scope() as query_id:
        assert bt.cancel_query(query_id) is True
        ds.collect()
    assert bt.running_queries() == []
    # The same query outside the cancelled scope is untouched.
    assert ds.collect().num_rows == 49_999


def test_cancelling_an_unknown_or_finished_id_reports_false():
    assert bt.cancel_query("q-never-registered") is False
    with query_scope() as query_id:
        pass
    assert bt.cancel_query(query_id) is False
