"""A float predicate pushed into the Parquet decode keeps every row DuckDB keeps.

`bc-io`'s row filter drops rows *during* decode, so it must never drop one the engine's
`Filter` would keep. Floats are the hard case: the decoder compares in IEEE `totalOrder`, where
`-0.0 < 0.0` and a sign-bit NaN sorts below `-inf`, while the engine -- like DuckDB -- treats
both zeros as one value and every NaN as the greatest. The row filter therefore keeps every
row the two orders could disagree on, and this file holds the result to DuckDB on data made of
exactly those rows.

The table is large enough (above the row filter's 200,000-row floor) and each predicate
selective enough (under its 50% ceiling) for the filter to be installed, and the payload column
makes it pay: its decode is what the filter skips for rejected rows. The Rust side pins the mask
itself against the engine's own canonicalization
(`row_filter::tests::a_float_mask_keeps_every_row_the_engine_keeps`); this is the end-to-end
half.
"""

from __future__ import annotations

import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_ROWS = 300_000
_NEGATIVE_NAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8_0000_0000_0001))[0]


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    rng = np.random.default_rng(11)
    f = rng.random(_ROWS)
    specials = np.array([0.0, -0.0, np.nan, _NEGATIVE_NAN, np.inf, -np.inf, 0.05, 0.07])
    at = rng.choice(_ROWS, 4_000, replace=False)
    f[at] = specials[np.arange(len(at)) % len(specials)]
    table = pa.table(
        {
            "f": pa.array(f),
            "f32": pa.array(f.astype("float32")),
            "q": rng.integers(0, 50, _ROWS),
            "payload": pa.array(rng.integers(0, 10**9, _ROWS).astype(str)),
        }
    )
    out = tmp_path_factory.mktemp("float_rf") / "t.parquet"
    pq.write_table(table, out, row_group_size=50_000)
    return str(out)


_PREDICATES = [
    ("f >= 0.0 AND f < 0.1", (bt.col("f") >= 0.0) & (bt.col("f") < 0.1)),
    ("f = 0.0", bt.col("f") == 0.0),
    ("f <= -0.0", bt.col("f") <= -0.0),
    ("f > 0.9", bt.col("f") > 0.9),
    ("f < 0.02", bt.col("f") < 0.02),
    ("f BETWEEN 0.05 AND 0.07", (bt.col("f") >= 0.05) & (bt.col("f") <= 0.07)),
    ("f > 0.95 OR q < 2", (bt.col("f") > 0.95) | (bt.col("q") < 2)),
    ("f32 > 0.9", bt.col("f32") > 0.9),
    ("f32 = 0.0", bt.col("f32") == 0.0),
]


@pytest.mark.parametrize("sql_pred, expr", _PREDICATES, ids=[p for p, _ in _PREDICATES])
def test_float_predicate_rows_match_duckdb(duck, path, sql_pred, expr):
    got = bt.read.parquet(path).filter(expr).select("payload", "q").collect()
    sql = f"SELECT payload, q FROM read_parquet('{path}', can_have_nan=true) WHERE {sql_pred}"
    assert_same(got, duck.sql(sql))


def test_the_disputed_rows_are_present_in_the_fixture(duck, path):
    """Positive control on the data: without NaNs and signed zeros the test proves nothing."""
    counts = duck.sql(
        f"SELECT count(*) FILTER (WHERE isnan(f)), count(*) FILTER (WHERE f = 0.0) "
        f"FROM read_parquet('{path}')"
    ).fetchone()
    assert counts[0] >= 500 and counts[1] >= 500
