"""SQL identifiers are case-insensitive; the relational layer is name-keyed.

``SELECT I FROM t`` is the same query as ``SELECT i FROM t`` in DuckDB, Postgres, Spark
and every warehouse a query is ported from — most of which upper-case their DDL. The
`Dataset` underneath is keyed by exact name, so an unquoted identifier typed in another
case raised "unknown column", which is most of a ported query.

Names are folded onto the relation's spelling before anything reads them, and the *output*
name is the relation's, which is what DuckDB reports (``SELECT I FROM t`` answers a column
called ``i``). An explicit alias still wins, and a relation genuinely carrying two columns
that differ only in case is left alone rather than silently resolved to one of them.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1, 2, 3], pa.int64()),
            "s": pa.array(["a", "b", "c"], pa.string()),
            "grp": pa.array(["x", "x", "y"], pa.string()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT I FROM t",
        "SELECT i FROM T",
        "SELECT I, S FROM T",
        "SELECT * FROM t WHERE I > 1",
        "SELECT GRP, count(*) AS c FROM t GROUP BY GRP",
        "SELECT UPPER(S) AS u FROM t",
        "SELECT i FROM T x WHERE X.I > 1",
        "SELECT Grp, sum(I) AS s FROM t GROUP BY Grp HAVING sum(I) > 1",
        "SELECT I AS X FROM t ORDER BY X",
        "SELECT count(*) AS c FROM T a JOIN T b ON a.I = b.I",
    ],
)
def test_an_identifier_resolves_regardless_of_case(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_the_output_name_is_the_relations_spelling_not_the_querys():
    """DuckDB answers `SELECT I FROM t` with a column named `i`; so must this."""
    table = _table()
    assert bt.sql("SELECT I FROM t", t=table).columns == ["i"]
    assert bt.sql("SELECT I AS X FROM t", t=table).columns == ["X"]


def test_an_ambiguous_fold_is_left_alone():
    """Two columns differing only in case are both real; neither may absorb the other."""
    table = pa.table({"id": pa.array([1]), "ID": pa.array([2])})
    got = bt.sql("SELECT id, ID FROM t", t=table).collect().to_pydict()
    assert got == {"id": [1], "ID": [2]}
