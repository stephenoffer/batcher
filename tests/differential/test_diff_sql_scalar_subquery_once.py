"""One uncorrelated scalar subquery written twice in a statement is evaluated once.

Evaluated per occurrence, a float aggregate could round its last bit differently from one run
to the next, so the "same" literal appeared as two values; TPC-DS q14's three `HAVING`s over
`avg_sales` then stopped being structurally equal and its ROLLUP levels lost their shared
aggregate. These cases pin that the memo is keyed by the bound relation, not the text alone:
the same text under a different `WITH` binding is a different subquery.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher._sql.parser.subquery.uncorrelated as scalar_mod
from _harness import assert_same


@pytest.fixture
def sess(duck):
    t = pa.table({"g": [1, 1, 2, 2, 3], "x": [0.1, 0.2, 0.3, 0.7, 1.1]})
    duck.register("t", t)
    s = bt.Session()
    s.register("t", t)
    return s


@pytest.fixture
def evaluations(monkeypatch):
    calls: list[str] = []
    real = scalar_mod._evaluate_scalar_subquery

    def counting(tr, node):
        calls.append(node.sql())
        return real(tr, node)

    monkeypatch.setattr(scalar_mod, "_evaluate_scalar_subquery", counting)
    return calls


def test_repeated_subquery_runs_once_and_matches_duckdb(duck, sess, evaluations):
    sql = """
        SELECT g, sum(x) AS s FROM t GROUP BY g HAVING sum(x) > (SELECT avg(x) FROM t)
        UNION ALL
        SELECT g, max(x) AS s FROM t GROUP BY g HAVING max(x) > (SELECT avg(x) FROM t)
    """
    out = sess.sql(sql).collect()
    assert len(evaluations) == 1, evaluations
    assert_same(out, duck.sql(sql))


def test_same_text_under_another_binding_runs_again(duck, sess, evaluations):
    sql = """
        WITH t AS (SELECT g, x * 10 AS x FROM t)
        SELECT g FROM t WHERE x > (SELECT avg(x) FROM t)
    """
    plain = "SELECT g FROM t WHERE x > (SELECT avg(x) FROM t)"
    assert_same(sess.sql(sql).collect(), duck.sql(sql))
    assert_same(sess.sql(plain).collect(), duck.sql(plain))
    assert len(evaluations) == 2, "two statements, two evaluations"


def test_distinct_subqueries_each_run(duck, sess, evaluations):
    sql = """
        SELECT g FROM t
        WHERE x > (SELECT min(x) FROM t) AND x < (SELECT max(x) FROM t)
    """
    assert_same(sess.sql(sql).collect(), duck.sql(sql))
    assert len(evaluations) == 2
