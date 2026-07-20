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
