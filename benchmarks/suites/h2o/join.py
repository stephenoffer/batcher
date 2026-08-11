"""H2O.ai db-benchmark, join task — all five of its questions.

The queries are the benchmark's own, taken from its ``duckdb/join-duckdb.R`` solution. One
LHS ``x`` is joined against three RHS tables whose sizes span six orders of magnitude —
``small`` (N/1e6 rows, on an integer key), ``medium`` (N/1e3), and ``big`` (N) — plus one
left outer join and one join on a string key. Because the generator gives the RHS 10% of
keys the LHS does not have and vice versa, an inner join genuinely drops rows and the outer
join genuinely produces nulls; a result that matches on row count but not on nulls is a real
disagreement, not a shape difference.

This is the suite's most direct test of build-side selection: the same query shape at three
RHS sizes is exactly the decision a cost model gets wrong when its cardinality estimate is
stale, and the reason the join task is cited as often as the groupby one.
"""

from __future__ import annotations

from registry import suite

join = suite("h2o-join", dataset="h2o-join")

# Keyed by the benchmark's own question numbering; the comment is its published description.
QUERIES: dict[str, str] = {
    # q1: small inner on int
    "h2o-join-q1": "SELECT x.*, small.id4 AS small_id4, v2 FROM x JOIN small USING (id1)",
    # q2: medium inner on int
    "h2o-join-q2": (
        "SELECT x.*, medium.id1 AS medium_id1, medium.id4 AS medium_id4, "
        "medium.id5 AS medium_id5, v2 FROM x JOIN medium USING (id2)"
    ),
    # q3: medium outer on int
    "h2o-join-q3": (
        "SELECT x.*, medium.id1 AS medium_id1, medium.id4 AS medium_id4, "
        "medium.id5 AS medium_id5, v2 FROM x LEFT JOIN medium USING (id2)"
    ),
    # q4: medium inner on factor
    "h2o-join-q4": (
        "SELECT x.*, medium.id1 AS medium_id1, medium.id2 AS medium_id2, "
        "medium.id4 AS medium_id4, v2 FROM x JOIN medium USING (id5)"
    ),
    # q5: big inner on int
    "h2o-join-q5": (
        "SELECT x.*, big.id1 AS big_id1, big.id2 AS big_id2, big.id4 AS big_id4, "
        "big.id5 AS big_id5, big.id6 AS big_id6, v2 FROM x JOIN big USING (id3)"
    ),
}

for _name, _query in QUERIES.items():
    join.sql(_name, _query)
