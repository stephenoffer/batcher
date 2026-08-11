"""Differential tests for factoring a conjunct out of an `OR` nested inside an `AND`.

`factor_common_conjuncts` rewrites `(A AND X) OR (A AND Y)` to `A AND (X OR Y)` so that an
equi-join condition hidden inside a disjunction is exposed to join-key derivation. It used
to fire only on a predicate that was *itself* a top-level `OR`; extending it to every
conjunct is what makes TPC-DS q13 and q48 plan as joins rather than as a chain of cartesian
products.

Because the rewrite now reaches many more predicates, its semantics matter more: these pin
the rewritten result against DuckDB across the shapes where boolean algebra is easy to get
subtly wrong — nulls under three-valued logic, a branch fully implied by the common
conjuncts, and a disjunction that must *not* be factored at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# A miniature of the TPC-DS q13 shape: a fact table whose join keys to two dimensions
# appear only inside disjunctions, plus nulls in both the keys and the filtered measures.
SALES = pa.table(
    {
        "s_dim1": [1, 2, 3, 1, 2, None, 3, 1],
        "s_dim2": [10, 20, 10, 30, None, 20, 30, 10],
        "price": [50.0, 120.0, 175.0, 90.0, 60.0, 110.0, None, 200.0],
        "profit": [120.0, 180.0, 60.0, 250.0, None, 140.0, 90.0, 175.0],
    }
)
DIM1 = pa.table({"d1_key": [1, 2, 3], "status": ["M", "S", "W"], "cnt": [3, 1, 1]})
DIM2 = pa.table(
    {"d2_key": [10, 20, 30], "state": ["TX", "OR", "VA"], "country": ["US", "US", "CA"]}
)


@pytest.fixture
def registered(duck):
    duck.register("sales", SALES)
    duck.register("dim1", DIM1)
    duck.register("dim2", DIM2)


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # The q13 shape itself: two disjunctions, each carrying the equi-key for one
        # dimension in every branch, under a top-level AND.
        """
        SELECT avg(price) AS a1, sum(profit) AS a2 FROM sales, dim1, dim2
        WHERE ((s_dim1 = d1_key AND status = 'M' AND price BETWEEN 100 AND 150 AND cnt = 3)
            OR (s_dim1 = d1_key AND status = 'S' AND price BETWEEN 50 AND 100 AND cnt = 1)
            OR (s_dim1 = d1_key AND status = 'W' AND price BETWEEN 150 AND 200 AND cnt = 1))
          AND ((s_dim2 = d2_key AND country = 'US' AND profit BETWEEN 100 AND 200)
            OR (s_dim2 = d2_key AND country = 'US' AND profit BETWEEN 150 AND 300))
        """,
        # One disjunction plus an ordinary conjunct beside it.
        """
        SELECT count(*) AS n FROM sales, dim1
        WHERE profit > 50
          AND ((s_dim1 = d1_key AND status = 'M') OR (s_dim1 = d1_key AND cnt = 1))
        """,
        # A branch entirely implied by the common conjuncts: the residual OR vanishes, so
        # the predicate must reduce to the join key alone, not to the key AND one branch.
        """
        SELECT count(*) AS n FROM sales, dim1
        WHERE (s_dim1 = d1_key AND status = 'M') OR (s_dim1 = d1_key)
        """,
        # Nothing shared by both branches — the rule must decline and the answer stand.
        """
        SELECT count(*) AS n FROM sales, dim1
        WHERE (s_dim1 = d1_key AND status = 'M') OR (cnt = 1 AND price > 100)
        """,
        # Nulls inside the factored conjunct: three-valued logic makes `A AND (X OR Y)`
        # equal to `(A AND X) OR (A AND Y)` only if NULL propagates identically.
        """
        SELECT count(*) AS n FROM sales
        WHERE (price IS NULL AND profit > 100) OR (price IS NULL AND profit < 100)
        """,
        """
        SELECT count(*) AS n FROM sales
        WHERE s_dim2 > 5 AND ((price > 100 AND profit IS NULL) OR (price > 100 AND s_dim1 = 2))
        """,
        # A LEFT join, where pulling a predicate out of the disjunction must not turn a
        # null-extended row into a match.
        """
        SELECT count(*) AS n FROM sales LEFT JOIN dim1 ON s_dim1 = d1_key
        WHERE (status = 'M' AND cnt = 3) OR (status = 'M' AND price > 100)
        """,
    ],
)
def test_factored_predicate_matches_duckdb(duck, registered, query):
    out = bt.sql(query, sales=SALES, dim1=DIM1, dim2=DIM2).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_nested_disjunction_does_not_plan_a_cartesian_product(registered):
    """The point of the rewrite: the join key must reach the plan, not a cross join.

    Without it the estimate is the product of the three inputs. The assertion is on the
    planner's own estimate rather than on wall-clock, because at benchmark scale the
    failure is not slowness — TPC-DS q13 estimated 1.2e16 rows and the process was killed.
    """
    query = """
        SELECT count(*) AS n FROM sales, dim1, dim2
        WHERE ((s_dim1 = d1_key AND status = 'M') OR (s_dim1 = d1_key AND cnt = 1))
          AND ((s_dim2 = d2_key AND country = 'US') OR (s_dim2 = d2_key AND state = 'TX'))
    """
    plan = bt.sql(query, sales=SALES, dim1=DIM1, dim2=DIM2).explain()
    cartesian = SALES.num_rows * DIM1.num_rows * DIM2.num_rows
    estimates = [
        int(part.split("est≈")[1].split()[0].replace(",", ""))
        for part in plan.splitlines()
        if "est≈" in part
    ]
    assert max(estimates) < cartesian
