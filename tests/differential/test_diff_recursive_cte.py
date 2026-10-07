"""`WITH RECURSIVE` vs DuckDB.

A recursive CTE is `anchor UNION [ALL] recursive-term`, evaluated to a fixpoint: run the
anchor, then repeatedly run the recursive term against *only the rows the last iteration
produced*, until an iteration yields nothing. Previously any self-reference raised
``unknown table``.

The `UNION` (distinct) form is the one that needs care: it is a *set* fixpoint, so rows
already derived must not be fed forward or a term that keeps re-deriving them never
terminates. `UNION ALL` has no such dedup and relies on its own stop predicate — which is
why the evaluation is bounded and raises rather than hanging.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


def _norm(d):
    n = len(next(iter(d.values()))) if d else 0
    return sorted([tuple(c[i] for c in d.values()) for i in range(n)], key=str)


@pytest.fixture
def edges(duck):
    # 1 → {2,3}, 2 → 4, 3 → 4, 4 → 5. Node 4 is reachable two ways, so a set fixpoint
    # must not emit it twice.
    table = pa.table({"src": [1, 1, 2, 3, 4], "dst": [2, 3, 4, 4, 5]})
    duck.register("edges", table)
    return table


@pytest.fixture
def one(duck):
    table = pa.table({"x": [1]})
    duck.register("one", table)
    return table


@pytest.mark.differential
@pytest.mark.parametrize(
    "body",
    [
        "SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 5",
        "SELECT 1 UNION ALL SELECT n * 2 FROM c WHERE n < 50",
        "SELECT 1 UNION SELECT n + 1 FROM c WHERE n < 4",
        # Degenerate: the term keeps re-deriving a row already in the set. Only the
        # distinct fixpoint's "new rows only" rule makes this terminate.
        "SELECT 1 UNION SELECT 1 FROM c WHERE n < 3",
        # Anchor produces several rows.
        "SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT n + 10 FROM c WHERE n < 3",
    ],
)
def test_recursive_cte_counters(duck, one, body):
    """Arithmetic recursions, both UNION and UNION ALL."""
    query = f"WITH RECURSIVE c(n) AS ({body}) SELECT n FROM c"
    got = bt.sql(query, one=one).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_recursive_cte_graph_reachability(duck, edges):
    """The real use case: transitive closure over a graph, joining back to the CTE."""
    query = (
        "WITH RECURSIVE reach(node) AS ("
        "  SELECT 1 UNION SELECT e.dst FROM edges e JOIN reach r ON e.src = r.node"
        ") SELECT node FROM reach"
    )
    got = bt.sql(query, edges=edges).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_recursive_cte_empty_anchor(duck, one):
    """An anchor matching nothing yields an empty relation, not an error or a hang."""
    query = (
        "WITH RECURSIVE c(n) AS (SELECT 1 WHERE 1 = 0 UNION ALL SELECT n + 1 FROM c) "
        "SELECT n FROM c"
    )
    got = bt.sql(query, one=one).collect()
    assert got.num_rows == 0
    assert _norm(got.to_pydict()) == _norm(duck.sql(query).to_arrow_table().to_pydict())


@pytest.mark.differential
def test_non_recursive_cte_in_a_recursive_block(duck, one):
    """`WITH RECURSIVE` marks the block, not each CTE — plain CTEs beside it still work."""
    query = (
        "WITH RECURSIVE plain AS (SELECT 7 AS v), "
        "c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 3) "
        "SELECT n, (SELECT v FROM plain) AS v FROM c"
    )
    got = bt.sql(query, one=one).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


def test_non_terminating_recursion_raises(one):
    """A missing stop condition must fail loudly rather than hang forever."""
    query = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT n FROM c"
    with pytest.raises(NotImplementedError, match="did not terminate"):
        bt.sql(query, one=one).collect()


def test_recursive_reference_in_the_anchor_rejects(one):
    """The first branch is the anchor and cannot reference the CTE."""
    query = "WITH RECURSIVE c(n) AS (SELECT n FROM c UNION ALL SELECT 1) SELECT n FROM c"
    with pytest.raises(NotImplementedError, match="anchor"):
        bt.sql(query, one=one).collect()


@pytest.mark.differential
@pytest.mark.parametrize(
    "body",
    [
        # The term re-derives a row holding a NULL. The set fixpoint compared rows with a
        # join, which never matches NULL, so the row was "new" on every iteration and the
        # recursion ran to the cap instead of stopping after one step.
        "SELECT 1, CAST(NULL AS BIGINT) UNION SELECT n, m FROM c",
        "SELECT CAST(NULL AS BIGINT), CAST(NULL AS BIGINT) UNION SELECT n, m FROM c",
        # A NULL-holding row alongside a terminating counter.
        "SELECT 1, CAST(NULL AS BIGINT) UNION SELECT least(n + 1, 3), m FROM c",
        # Duplicates inside one iteration collapse too.
        "SELECT 1, CAST(NULL AS BIGINT) UNION ALL SELECT 1, CAST(NULL AS BIGINT) "
        "UNION SELECT n, m FROM c",
    ],
)
def test_recursive_union_with_null_rows_terminates(duck, one, body):
    """A ``UNION`` recursion over rows holding NULLs reaches its fixpoint, as DuckDB's does."""
    query = f"WITH RECURSIVE c(n, m) AS ({body}) SELECT n, m FROM c"
    got = bt.Session(max_recursion=50).sql(query, one=one).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


def test_max_recursion_moves_the_cap(one):
    """``Session(max_recursion=n)`` refuses at n iterations, and the error says how far it got."""
    query = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT n FROM c"
    with pytest.raises(bt.SQLUnsupportedError, match="within 5 iterations") as raised:
        bt.Session(max_recursion=5).sql(query, one=one).collect()
    assert "6 rows accumulated" in str(raised.value)
    assert "max_recursion" in raised.value.hint
    # The same eight-step recursion is refused under a cap of 5 and runs under a cap of 10,
    # so the setting is what decides, not the query.
    eight = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 8) "
    with pytest.raises(bt.SQLUnsupportedError, match="within 5 iterations"):
        bt.Session(max_recursion=5).sql(eight + "SELECT n FROM c", one=one)
    got = bt.Session(max_recursion=10).sql(eight + "SELECT count(*) AS k FROM c").to_pydict()
    assert got == {"k": [8]}


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_max_recursion_rejects_a_non_positive_int(bad):
    with pytest.raises(bt.PlanError, match="max_recursion"):
        bt.Session(max_recursion=bad)
