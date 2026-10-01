"""JOB-shaped join regions match DuckDB after implied-key pruning and key-range sinking.

Join reordering now keeps one key pair per equality a side does not already imply, and the key
range a join derives for a side sinks beneath that side's own filters. Both change the plan
and neither may change a row. The fixtures are the shapes that exercise them: a clique of
pairwise equalities on one movie key, written the way the Join Order Benchmark writes it, over
fact tables large enough to shard (more than 65,536 rows), with NULL keys, duplicated keys, an
empty side, and a narrow link table whose key range is what gets sunk.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_N = 70_000


def _tables(with_nulls: bool = True) -> dict[str, pa.Table]:
    ids = list(range(_N))
    mi_ids = [None if with_nulls and i % 97 == 0 else i % _N for i in range(3 * _N)]
    return {
        "t": pa.table({"id": ids, "year": [1900 + i % 120 for i in ids]}),
        "mi": pa.table(
            {
                "movie_id": mi_ids,
                "info": [("USA:" if i % 7 == 0 else "Bel:") + str(i % 50) for i in range(3 * _N)],
            }
        ),
        "mk": pa.table(
            {
                "movie_id": [i % 9_000 for i in range(2 * _N)],
                "keyword_id": [i % 300 for i in range(2 * _N)],
            }
        ),
        "mc": pa.table(
            {
                "movie_id": [(i * 7) % _N for i in range(_N)],
                "company_id": [i % 1_000 for i in range(_N)],
            }
        ),
        # The narrow link table: its movie ids span a sliver of the id range, which is the key
        # range `runtime_join_filter` hands to every other side of the region.
        "ml": pa.table(
            {
                "movie_id": [500 + (i % 2_000) for i in range(6_000)],
                "link": [i % 4 for i in range(6_000)],
            }
        ),
        "k": pa.table({"id": list(range(300)), "keyword": [f"kw{i}" for i in range(300)]}),
        "empty": pa.table({"movie_id": pa.array([], pa.int64()), "x": pa.array([], pa.int64())}),
    }


def _run(sql: str, duck, tables: dict[str, pa.Table]) -> None:
    s = bt.Session()
    for name, table in tables.items():
        s.register(name, table)
        duck.register(name, table)
    assert_same(s.sql(sql).collect(), duck.sql(sql))


_CLIQUE = """
SELECT MIN(t.year) AS y, MIN(mi.info) AS info, COUNT(*) AS n
FROM t, mi, mk, mc
WHERE t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id
  AND mi.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mk.movie_id = mc.movie_id
  AND t.year > 1990 AND mi.info LIKE 'USA:%'
"""

_WITH_LINK = """
SELECT ml.link, COUNT(*) AS n, MIN(k.keyword) AS kw, MAX(t.year) AS y
FROM t, mi, mk, ml, k
WHERE t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = ml.movie_id
  AND mi.movie_id = mk.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = ml.movie_id
  AND mk.keyword_id = k.id
  AND mi.info IN ('USA:1', 'USA:8', 'Bel:3') AND t.year BETWEEN 1950 AND 2010
GROUP BY ml.link
"""


@pytest.mark.parametrize("with_nulls", [True, False])
def test_a_key_clique_matches_duckdb(duck, with_nulls):
    _run(_CLIQUE, duck, _tables(with_nulls))


@pytest.mark.parametrize("with_nulls", [True, False])
def test_a_narrow_link_range_sunk_beneath_string_filters_matches_duckdb(duck, with_nulls):
    _run(_WITH_LINK, duck, _tables(with_nulls))


def test_rows_not_only_aggregates_match_duckdb(duck):
    # Every joined row, duplicates included, so a dropped key pair that let extra rows through
    # (or a sunk range that removed a match) shows up as a multiset difference.
    sql = """
    SELECT t.id, mi.info, mk.keyword_id, ml.link
    FROM t, mi, mk, ml
    WHERE t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = ml.movie_id
      AND mi.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND mi.movie_id = ml.movie_id
      AND mi.info LIKE '%:4%'
    """
    _run(sql, duck, _tables())


def test_a_clique_with_an_empty_member_is_empty(duck):
    sql = """
    SELECT COUNT(*) AS n, MIN(t.year) AS y
    FROM t, mi, mk, empty
    WHERE t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = empty.movie_id
      AND mi.movie_id = mk.movie_id AND mi.movie_id = empty.movie_id
      AND mk.movie_id = empty.movie_id
    """
    _run(sql, duck, _tables())


def test_one_row_sides_match_duckdb(duck):
    tables = _tables()
    tables["ml"] = pa.table({"movie_id": [700], "link": [1]})
    _run(_WITH_LINK, duck, tables)
