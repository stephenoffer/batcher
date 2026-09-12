"""A `GROUP BY` over two ORDERED relations concatenated must not split a key into two groups.

`agg_par::key_disjoint_runs` cuts an ordered relation into runs it can aggregate and then glue
with `concat_disjoint`, which skips the regroup entirely. The cut used to be legal against the
*next morsel* rather than against everything after it — the same question on a globally ordered
relation, a different one on a concatenation of two. A `UNION`'s left side ascends and cuts
freely, its right side restarts at the bottom of the key space, and every left run then overlaps
the run holding the right side. `concat_disjoint` emits each shared key once per run.

`INTERSECT` and `EXCEPT` lower to exactly that shape (`union → GROUP BY k` with `bool_or` tags),
so the bug reached them by construction: over 6,000,000 sorted rows against 1,500,000,
`INTERSECT` returned **4,906 rows where the answer is 1,472,588** and `EXCEPT` returned
1,467,682 where the answer is 0 — the two wrong answers summing to the right distinct count,
which is the signature of one key landing in two groups.

The inputs here are large and sorted on purpose. The path is reached only when a relation is big
enough to be cut into runs, so a small fixture exercises none of it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# `key_disjoint_runs` only cuts once a relation yields enough runs per worker, so the size is
# load-bearing rather than incidental: at 2,000,000 x 500,000 it declines, every assertion below
# passes against the *broken* engine, and the test proves nothing. Verified by reintroducing the
# defect and re-running — these figures fail, those do not.
_LEFT_ROWS = 6_000_000
_KEYS = 1_500_000


@pytest.fixture(scope="module")
def ordered_pair() -> tuple[pa.Table, pa.Table]:
    rng = np.random.default_rng(20260912)
    left = np.sort(rng.integers(1, _KEYS + 1, _LEFT_ROWS))
    return (
        pa.table({"k": pa.array(left, type=pa.int64())}),
        pa.table({"k": pa.array(np.arange(1, _KEYS + 1), type=pa.int64())}),
    )


@pytest.mark.parametrize("op", ["INTERSECT", "EXCEPT", "UNION"])
def test_a_set_op_over_a_sorted_left_input_matches_duckdb(duck, ordered_pair, op) -> None:
    left, right = ordered_pair
    duck.register("l", left)
    duck.register("r", right)
    sql = f"SELECT COUNT(*) AS n FROM (SELECT k FROM l {op} SELECT k FROM r) u"
    session = bt.Session()
    session.register("l", left)
    session.register("r", right)
    assert_same(session.sql(sql).collect(), duck.sql(sql))


def test_a_group_by_over_two_ordered_relations_emits_one_row_per_key(duck, ordered_pair) -> None:
    """The general form — the set ops above are one way to produce this shape, not the only one.

    Asserted as a group *count* as well as against DuckDB, because a duplicated key is a wrong
    row count and that is the thing to name when it regresses.
    """
    left, right = ordered_pair
    duck.register("l", left)
    duck.register("r", right)
    sql = (
        "SELECT COUNT(*) AS groups FROM ("
        "  SELECT k, COUNT(*) AS c FROM ("
        "    SELECT k FROM l UNION ALL SELECT k FROM r"
        "  ) q GROUP BY k"
        ") z"
    )
    session = bt.Session()
    session.register("l", left)
    session.register("r", right)
    got = session.sql(sql).collect()
    assert got.to_pydict()["groups"] == [_KEYS]
    assert_same(got, duck.sql(sql))
