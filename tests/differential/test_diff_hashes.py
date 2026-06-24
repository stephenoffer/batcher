"""Differential coverage for the crypto/non-crypto string hash functions.

`md5`/`sha1`/`sha256` are checked against DuckDB (which has the same functions);
`crc32`/`xxhash64` have no DuckDB equivalent, so they are pinned to known fixed
vectors (the oracle is the published algorithm, documented per testing.md).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _strings():
    return pa.table({"s": pa.array(["abc", "hello world", "", "Ünïcödé", None])})


@pytest.mark.parametrize("fn", ["md5", "sha1", "sha256"])
def test_crypto_hash_matches_duckdb(duck, fn):
    from conftest import assert_same

    duck.register("t", _strings())
    out = bt.from_arrow(_strings()).select(h=getattr(col("s").str, fn)()).collect()
    assert_same(out, duck.sql(f"SELECT {fn}(s) AS h FROM t"))


def test_crc32_and_xxhash_fixed_vectors():
    # No DuckDB oracle — pin to the published algorithm outputs (null → null).
    out = (
        bt.from_arrow(_strings())
        .select(crc=col("s").str.crc32(), xx=col("s").str.xxhash64())
        .collect()
        .to_pydict()
    )
    # CRC-32/IEEE of "abc" is 0x352441C2 = 891568578; empty string is 0.
    assert out["crc"][0] == 891568578
    assert out["crc"][2] == 0
    assert out["crc"][4] is None
    # xxHash64 (seed 0) of the empty string is the well-known 0xEF46DB3751D8E999.
    assert out["xx"][2] == -1205034819632174695  # 0xEF46DB3751D8E999 as i64
    assert out["xx"][4] is None
    # Deterministic: same input → same digest.
    again = bt.from_arrow(_strings()).select(xx=col("s").str.xxhash64()).collect().to_pydict()
    assert again["xx"] == out["xx"]
