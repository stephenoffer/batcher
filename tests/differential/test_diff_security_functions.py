"""`mask` / `hmac_sha256` / `aes_encrypt` / `aes_decrypt` cross-checked against DuckDB.

DuckDB has no masking or AES function, so the oracle cannot be "call DuckDB's version
of this". Instead each test states the function's *defining property* as a DuckDB query
Batcher must match:

* `mask` is `repeat(char, n - k) || right(s, k)` — a composition DuckDB does have.
* `aes_decrypt ∘ aes_encrypt` is the identity, so the oracle is the plaintext column.
* `hmac_sha256` is an equivalence-preserving map: it partitions rows exactly as the
  plaintext does, so its `GROUP BY` counts must equal the plaintext's.

That makes the tests real differential tests — a Batcher-only implementation of the
property would still have to agree with an independent engine on the outcome — rather
than assertions against hardcoded digests, which only pin the implementation to itself.
The known-answer vectors (RFC 4231 etc.) live in the Rust unit tests.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential

# A test key, as 64 hex characters. Never a real key: it is checked into the repo.
KEY = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"


def _cards(duck):
    t = pa.table(
        {
            "card": pa.array(
                ["4111111111111234", "5500000000000004", "abc", "", None], type=pa.string()
            )
        }
    )
    duck.register("cards", t)
    return bt.from_arrow(t)


def test_mask_show_last_matches_duckdb_composition(duck):
    """Revealing the last 4 == repeat('X', len-4) || right(s, 4), where len > 4."""
    ds = _cards(duck)
    got = ds.select(m=bt.mask(col("card"), show_last=4)).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT CASE WHEN length(card) <= 4 THEN card "
            "ELSE repeat('X', length(card) - 4) || right(card, 4) END AS m FROM cards"
        ),
    )


def test_mask_show_first_matches_duckdb_composition(duck):
    ds = _cards(duck)
    got = ds.select(m=bt.mask(col("card"), show_first=2, char="*")).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT CASE WHEN length(card) <= 2 THEN card "
            "ELSE substr(card, 1, 2) || repeat('*', length(card) - 2) END AS m FROM cards"
        ),
    )


def test_mask_preserves_character_length(duck):
    """A masked value is the same length as its input — the property a UI depends on."""
    ds = _cards(duck)
    got = ds.select(n=bt.mask(col("card")).str.len()).to_arrow()
    assert_same(got, duck.sql("SELECT length(card) AS n FROM cards"))


def test_mask_of_a_short_value_is_the_value(duck):
    """Overlapping reveal windows must not silently produce cleartext-shaped garbage."""
    ds = _cards(duck)
    got = ds.select(m=bt.mask(col("card"), show_first=3, show_last=3)).to_arrow()
    assert_same(
        got,
        duck.sql(
            "SELECT CASE WHEN length(card) <= 6 THEN card "
            "ELSE substr(card, 1, 3) || repeat('X', length(card) - 6) || right(card, 3) END AS m "
            "FROM cards"
        ),
    )


def test_aes_round_trip_is_the_identity(duck):
    """decrypt(encrypt(x)) == x, including the null and the empty string."""
    ds = _cards(duck)
    got = ds.select(card=bt.aes_decrypt(bt.aes_encrypt(col("card"), KEY), KEY)).to_arrow()
    assert_same(got, duck.sql("SELECT card FROM cards"))


def test_aes_decrypt_under_a_wrong_key_is_null_not_an_error(duck):
    """A wrong key nulls the column rather than failing the query."""
    ds = _cards(duck)
    got = ds.select(card=bt.aes_decrypt(bt.aes_encrypt(col("card"), KEY), "f" * 64)).to_arrow()
    assert_same(got, duck.sql("SELECT NULL::VARCHAR AS card FROM cards"))


def test_aes_encrypt_preserves_distinctness(duck):
    """Encryption is injective: it neither merges nor splits distinct plaintexts."""
    ds = _cards(duck)
    got = ds.select(c=bt.aes_encrypt(col("card"), KEY)).agg(n=col("c").n_unique()).to_arrow()
    assert_same(got, duck.sql("SELECT COUNT(DISTINCT card) AS n FROM cards"))


def test_hmac_preserves_the_grouping_of_the_plaintext(duck):
    """Pseudonymization is equivalence-preserving, so GROUP BY counts are unchanged."""
    t = pa.table({"email": ["a@x.com", "a@x.com", "b@x.com", None, None]})
    duck.register("users", t)
    got = (
        bt.from_arrow(t)
        .select(p=bt.hmac_sha256(col("email"), key="s3cret"))
        .group_by("p")
        .agg(n=bt.count())
        .select("n")
        .to_arrow()
    )
    assert_same(got, duck.sql("SELECT COUNT(*) AS n FROM users GROUP BY email"))


def test_hmac_is_a_64_character_hex_digest(duck):
    t = pa.table({"email": ["a@x.com", None]})
    duck.register("users", t)
    got = bt.from_arrow(t).select(n=bt.hmac_sha256(col("email"), key="k").str.len()).to_arrow()
    assert_same(
        got, duck.sql("SELECT CASE WHEN email IS NULL THEN NULL ELSE 64 END AS n FROM users")
    )


def test_hmac_differs_from_the_unkeyed_digest(duck):
    """The key is actually mixed in — otherwise this is a rebranded `sha256`."""
    t = pa.table({"email": ["a@x.com"]})
    duck.register("users", t)
    got = (
        bt.from_arrow(t)
        .select(same=bt.hmac_sha256(col("email"), key="k") == col("email").str.sha256())
        .to_arrow()
    )
    assert_same(got, duck.sql("SELECT false AS same FROM users"))
