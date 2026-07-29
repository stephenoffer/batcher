"""A learned statistic may choose a plan. It may never decide which rows exist.

Kyber's moat is that statistics measured from one run improve the *plan* of the next. The
entire safety of that loop rests on one property: a stat is an input to a cost decision, and
a wrong stat must cost a query time, never correctness. `CLAUDE.md` names the failure mode --
"Core measures, Kyber decides" -- because a metadata path that quietly decides a *result*
passes every per-operator test while corrupting answers across queries.

It happened. `grouped_aggregate_columns` published a per-group `count(*)` upper bound of
`|child|` taken from the *estimated* row count, and `zonemap_prune_filter` folds a
`HAVING count(*) > n` whose bound cannot reach `n` into the empty relation. So after a
selective query taught the process-global hub that a scan yields ~1 row, an unrelated
aggregate over the same source with a *different* predicate returned nothing -- and a later,
less selective query silently restored the right answer.

The property under test is deliberately broad: run a query, run other queries, run it again,
and the answer must not move. It needs no oracle, and it is the only shape that can catch the
next version of this bug, wherever in the metadata loop it lands.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

_rng = np.random.default_rng(5)
_N = 300
TABLE = pa.table(
    {
        "id": pa.array(np.arange(_N, dtype=np.int64)),
        "k": pa.array(_rng.integers(0, 12, _N).astype(np.int64)),
        "s": pa.array(
            [None if i % 19 == 0 else str(x) for i, x in enumerate(_rng.choice(list("abcdef"), _N))]
        ),
    }
)

_GROUPED = "SELECT s, min(k) AS a FROM t WHERE id BETWEEN -5 AND 40 GROUP BY s HAVING count(*)"

#: Queries whose answer must never depend on what ran before them.
PROBES = {
    "having_gt6": f"{_GROUPED} > 6",
    "having_gt2": f"{_GROUPED} > 2",
    "having_gt0": f"{_GROUPED} > 0",
    "plain_group": "SELECT s, count(*) AS c FROM t WHERE id BETWEEN -5 AND 40 GROUP BY s",
    "filtered_count": "SELECT count(*) AS c FROM t WHERE id BETWEEN -5 AND 40",
}

#: Run between the probes. Each is *more selective* than the probes' own predicate, which is
#: what makes a learned row count small enough to look like a proof that no group is large.
POISONS = [
    "SELECT id FROM t WHERE id <= 0",
    "SELECT id FROM t WHERE id <= 3",
    "SELECT count(*) AS c FROM t WHERE id < 1",
    "SELECT k FROM t WHERE id = 0",
    "SELECT s, count(*) AS c FROM t WHERE id <= 1 GROUP BY s",
]


def _session():
    session = bt.Session()
    session.register("t", bt.from_arrow(TABLE))
    return session


def _answer(session, sql):
    """The result as an order-independent, null-safe sorted list of rows."""
    out = session.sql(sql).collect().to_pydict()
    if not out:
        return []
    rows = list(zip(*out.values(), strict=True))
    return sorted(rows, key=lambda row: [(v is None, "" if v is None else str(v)) for v in row])


@pytest.mark.parametrize("probe", sorted(PROBES))
def test_no_earlier_query_changes_a_later_answer(probe):
    sql = PROBES[probe]
    session = _session()
    before = _answer(session, sql)
    assert before, f"{probe} should return rows to begin with"

    for poison in POISONS:
        session.sql(poison).collect()
        assert _answer(session, sql) == before, (
            f"{probe} changed after running {poison!r} -- a statistic decided a result"
        )

    # The learned state is process-global, so a brand-new Session must be unaffected too.
    assert _answer(_session(), sql) == before, f"{probe} changed for a fresh Session"


def test_a_having_clause_survives_a_selective_neighbour(duck):
    """...and the answer that must not move is also the right one."""
    duck.register("t", TABLE)
    sql = PROBES["having_gt6"]
    session = _session()
    session.sql("SELECT id FROM t WHERE id <= 0").collect()
    got = _answer(session, sql)
    want = sorted(
        ((r[0], r[1]) for r in duck.sql(sql).fetchall()),
        key=lambda row: [(v is None, "" if v is None else str(v)) for v in row],
    )
    assert got == want
