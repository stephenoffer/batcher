"""An `EXISTS` under `OR` must not be joined to the FROM clause's cross product.

`EXISTS (…) OR …` cannot become a semi join, so `subquery.core._exists_marker` attaches a
boolean marker with a LEFT JOIN — and it does so *immediately*, against whatever the relation
is at that moment. Every other WHERE conjunct only accumulates into the residual the caller
filters with afterwards, so on the comma-join shape

    FROM a, b, c WHERE a.k = b.k AND c.k = a.k AND (EXISTS (…) OR …)

that moment was the bare **cross product** `a x b x c`: the marker was joined to it, and the
equalities that make it three ordinary joins were applied only after.

That is quadratic in the width of the FROM clause. TPC-DS q10 is exactly this shape and was
**OOM-killed on sf1 (371 MB)** where DuckDB answers in ~32 ms; bisected on that data, holding
the subquery fixed and adding one comma-joined table took it from 425 ms to 23,858 ms, and
three tables killed the process.

The fix promotes the `col = col` equalities ahead of the marker. These tests pin both halves:
the answer still matches DuckDB, **and** no operator ever sees the cross product. The second
assertion is the one that fails without the fix — with small inputs the old plan is merely
wasteful rather than fatal, so a correctness-only test would pass against the bug.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# Small enough that the cross product is survivable (the test must fail, not hang), large
# enough that it is unmistakable: 3 x 200 rows is 8,000,000 crossed against 200 joined.
_N = 200
_CROSS_PRODUCT_FLOOR = 10_000


def _tables(duck):
    a = pa.table({"ak": pa.array(range(_N), pa.int64()), "av": pa.array(range(_N), pa.int64())})
    b = pa.table({"bk": pa.array(range(_N), pa.int64()), "bv": pa.array(range(_N), pa.int64())})
    c = pa.table({"ck": pa.array(range(_N), pa.int64()), "cv": pa.array(range(_N), pa.int64())})
    # The correlated inner relation: every third `ak` has a match.
    s = pa.table({"sk": pa.array(range(0, _N, 3), pa.int64())})
    t = pa.table({"tk": pa.array(range(0, _N, 5), pa.int64())})
    for name, tbl in (("a", a), ("b", b), ("c", c), ("s", s), ("t", t)):
        duck.register(name, tbl)
    sess = bt.Session()
    for name, tbl in (("a", a), ("b", b), ("c", c), ("s", s), ("t", t)):
        sess.register(name, tbl)
    return sess


_SQL = """
SELECT count(*) AS n FROM a, b, c
WHERE a.ak = b.bk AND c.ck = a.ak
  AND (EXISTS (SELECT * FROM s WHERE s.sk = a.ak)
       OR EXISTS (SELECT * FROM t WHERE t.tk = a.ak))
"""

_SQL_SINGLE_MARKER = """
SELECT count(*) AS n FROM a, b, c
WHERE a.ak = b.bk AND c.ck = a.ak
  AND (EXISTS (SELECT * FROM s WHERE s.sk = a.ak) OR a.av < 0)
"""


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), [("or-of-two", _SQL), ("single", _SQL_SINGLE_MARKER)])
def test_exists_under_or_matches_duckdb(duck, label, sql):
    """The answer is DuckDB's, whichever way the marker is reached."""
    sess = _tables(duck)
    assert_same(sess.sql(sql).collect(), duck.sql(sql)), label


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), [("or-of-two", _SQL), ("single", _SQL_SINGLE_MARKER)])
def test_the_marker_is_not_joined_to_the_cross_product(duck, label, sql):
    """No operator may process the FROM clause's cross product.

    This is the regression itself. Three 200-row tables joined on a key produce 200 rows;
    crossed they produce 8,000,000, and the old plan built the marker's LEFT JOIN on top of
    that. Asserting on measured `rows_in` rather than on wall time keeps it deterministic —
    a timing threshold would be flaky on a shared box, and a correctness assertion alone
    cannot see the difference at all.
    """
    sess = _tables(duck)
    ds = sess.sql(sql)
    ds.collect()
    widest = max(int(r) for r in ds.stats().to_pandas()["rows_in"])
    assert widest < _CROSS_PRODUCT_FLOOR, (
        f"{label}: an operator saw {widest:,} rows for a {_N}-row join — "
        "the existence marker is being applied to the FROM clause's cross product"
    )
