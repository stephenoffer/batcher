"""`list_sum`/`list_avg` over integer lists must be exact, not routed via Float64.

Reducing an integer list through a Float64 view rounds every element to 53
significant bits, so `list_sum([2**53 + 1, 2])` returned `2**53 + 2` where the
true — and DuckDB's — answer is `2**53 + 3`. `list_avg` inherited the same lost
bit even though its *result* is legitimately a double. These cases pin the exact
integer accumulation against DuckDB, plus the null/empty/all-null edges.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

BIG = (1 << 53) + 1


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "id": pa.array([0, 1, 2, 3, 4, 5], type=pa.int64()),
            "l": pa.array(
                [
                    [BIG, 2],  # exactness above 2**53
                    [1, 2, 3],  # ordinary
                    [],  # empty row -> null
                    None,  # null row -> null
                    [None, None],  # all-null row -> null
                    [-5, 3, None],  # negatives, null element skipped
                ],
                type=pa.list_(pa.int64()),
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, list_avg(l) AS r FROM t",
        # `list_product` is a double in both engines — asserted here so a future
        # change to the integer path cannot silently alter it.
        "SELECT id, list_product(l) AS r FROM t",
    ],
)
def test_list_int_reductions_match_duckdb(duck, t, q):
    assert_same(
        bt.from_arrow(t).sql(q.replace(" FROM t", " FROM self")).to_arrow(),
        duck.sql(q),
    )


@pytest.mark.differential
def test_list_sum_matches_duckdb_exactly(duck, t):
    """`list_sum` equals DuckDB *exactly*, compared without a float round-trip.

    `assert_same` cannot express this case: DuckDB returns `list_sum` as
    `DECIMAL(38, 0)` and the harness coerces Decimal to float, which rounds
    `9007199254740995` to `...996` — the very bit under test. So compare the
    integers themselves. Both engines produce `2**53 + 3`; only the harness's
    comparison view is lossy, so this is a stricter check, not a weaker one.
    """
    out = bt.from_arrow(t).sql("SELECT list_sum(l) AS r FROM self").to_arrow()
    assert pa.types.is_integer(out.schema.field("r").type)
    expected = [
        None if v is None else int(v)
        for v in duck.sql("SELECT list_sum(l) AS r FROM t ORDER BY id")
        .to_arrow_table()
        .column("r")
        .to_pylist()
    ]
    assert expected[0] == BIG + 2, "sanity: DuckDB itself is exact here"
    assert out.column("r").to_pylist() == expected


@pytest.mark.differential
def test_list_avg_accumulates_exactly(t):
    """`list_avg` stays a double, but is computed from the exact integer total."""
    out = bt.from_arrow(t).sql("SELECT list_avg(l) AS r FROM self").to_arrow()
    assert pa.types.is_floating(out.schema.field("r").type)
    assert out.column("r").to_pylist()[0] == (BIG + 2) / 2
