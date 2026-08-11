"""H2O.ai db-benchmark, groupby task — all ten of its questions.

The queries are the benchmark's own, taken from its ``duckdb/groupby-duckdb.R`` solution so
every SQL engine here runs what the published leaderboard ran. They sweep the aggregation
space deliberately: single and compound keys, a key with 100 groups against one with N/100,
mean/sum/count, a median-and-stddev pair, a correlation, a top-2-per-group window, and
finally a group-by on all six key columns at once (the near-unique-key case that turns the
hash table into the whole dataset).

Batcher's advantage should show on the high-cardinality ones (q3, q6, q10), where the
group-by state is what dominates; a regression there is a regression in ``bc-runtime``'s
mergeable aggregate, not in the query.

One substitution in the SQL, stated rather than quietly applied: q6 uses ``median(v3)``
where the reference solution writes ``quantile_cont(v3, 0.5)``. They are the same function
in DuckDB, and ``median`` is the spelling every engine here has.

One known engine divergence, which the correctness gate reports rather than tolerates: q9's
``corr`` over a group with no variance in ``v1`` returns ``NaN`` in DuckDB and ``NULL`` in
Batcher (PostgreSQL's answer). That is a real difference in an aggregate's degenerate case,
not a rounding artifact. It cannot arise at the benchmark's own 1e7-row tier or above, where
every one of the 10,000 ``id2 x id4`` groups holds ~1,000 rows — only at a sub-tier smoke
scale such as ``--scale 0.01``, where a group can be small enough to be constant.
"""

from __future__ import annotations

from registry import suite

groupby = suite("h2o-groupby", dataset="h2o-groupby")

# Keyed by the benchmark's own question numbering; the comment is its published description.
QUERIES: dict[str, str] = {
    # q1: sum v1 by id1
    "h2o-gb-q1": "SELECT id1, sum(v1) AS v1 FROM x GROUP BY id1",
    # q2: sum v1 by id1:id2
    "h2o-gb-q2": "SELECT id1, id2, sum(v1) AS v1 FROM x GROUP BY id1, id2",
    # q3: sum v1 mean v3 by id3
    "h2o-gb-q3": "SELECT id3, sum(v1) AS v1, avg(v3) AS v3 FROM x GROUP BY id3",
    # q4: mean v1:v3 by id4
    "h2o-gb-q4": ("SELECT id4, avg(v1) AS v1, avg(v2) AS v2, avg(v3) AS v3 FROM x GROUP BY id4"),
    # q5: sum v1:v3 by id6
    "h2o-gb-q5": ("SELECT id6, sum(v1) AS v1, sum(v2) AS v2, sum(v3) AS v3 FROM x GROUP BY id6"),
    # q6: median v3 sd v3 by id4 id5
    "h2o-gb-q6": (
        "SELECT id4, id5, median(v3) AS median_v3, stddev(v3) AS sd_v3 FROM x GROUP BY id4, id5"
    ),
    # q7: max v1 - min v2 by id3
    "h2o-gb-q7": "SELECT id3, max(v1) - min(v2) AS range_v1_v2 FROM x GROUP BY id3",
    # q8: largest two v3 by id6
    "h2o-gb-q8": (
        "SELECT id6, v3 AS largest2_v3 FROM ("
        "  SELECT id6, v3, row_number() OVER (PARTITION BY id6 ORDER BY v3 DESC) AS order_v3"
        "  FROM x WHERE v3 IS NOT NULL) sub_query "
        "WHERE order_v3 <= 2"
    ),
    # q9: regression v1 v2 by id2 id4
    "h2o-gb-q9": ("SELECT id2, id4, pow(corr(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4"),
    # q10: sum v3 count by id1:id6
    "h2o-gb-q10": (
        "SELECT id1, id2, id3, id4, id5, id6, sum(v3) AS v3, count(*) AS count "
        "FROM x GROUP BY id1, id2, id3, id4, id5, id6"
    ),
}

for _name, _query in QUERIES.items():
    groupby.sql(_name, _query)
