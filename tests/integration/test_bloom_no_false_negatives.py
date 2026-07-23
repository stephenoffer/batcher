"""The bloom index must never report a present value as absent.

`zonemap_prune_filter` treats `BloomIndex.contains(v) is False` as a **proof** that no row
satisfies `col = v`, and replaces the whole relation with an empty one. So a single false
negative does not degrade a plan — it silently deletes rows.

That proof rests on an unverified cross-language agreement: the index is built in Rust
(`bc_py::build_column_bloom`, hashing Int64 as 8-byte little-endian and Utf8 as raw bytes,
with FNV-1a and double hashing) and probed by a *pure-Python* reimplementation
(`plan.bloom_index`). Nothing tested that the two agree. These tests do, over the value
shapes most likely to diverge: negative integers, the Int64 extremes, zero, non-ASCII and
empty strings.

They also pin the type-domain guard: an integer literal probed against a *string* column's
index would hash entirely different bytes and report a definitive — and wrong — absence.
"""

from __future__ import annotations

import random
import string

import pyarrow as pa
import pytest

from batcher.plan.bloom_index import BloomIndex

nat = pytest.importorskip("batcher._native", reason="native engine not built")

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _bloom_over(values: list, arrow_type: pa.DataType) -> BloomIndex:
    batch = pa.record_batch([pa.array(values, arrow_type)], names=["c"])
    raw = nat.build_column_bloom([batch], 0, max(1, len(values)))
    assert raw is not None, f"{arrow_type} should be indexable"
    index = BloomIndex.from_bytes(raw)
    assert index is not None, "the Python reader must parse what Rust wrote"
    return index


@pytest.mark.integration
def test_every_inserted_integer_is_reported_present():
    rng = random.Random(1234)
    values = [
        0,
        1,
        -1,
        _I64_MIN,
        _I64_MAX,
        *[rng.randint(_I64_MIN, _I64_MAX) for _ in range(500)],
    ]
    index = _bloom_over(values, pa.int64())
    for v in values:
        assert index.contains(v), f"false negative on {v} — pruning would delete its rows"


@pytest.mark.integration
def test_every_inserted_string_is_reported_present():
    rng = random.Random(99)
    values = [
        "",
        "a",
        "naïve",
        "日本語",
        "\x00embedded-nul",
        *["".join(rng.choices(string.printable, k=rng.randint(1, 40))) for _ in range(500)],
    ]
    index = _bloom_over(values, pa.utf8())
    for v in values:
        assert index.contains(v), f"false negative on {v!r} — pruning would delete its rows"


@pytest.mark.integration
def test_absent_values_are_mostly_reported_absent():
    # Not a correctness requirement (false positives are allowed), but a bloom that says
    # "present" for everything would make the whole index useless while passing the tests
    # above. 1% target FPR over 1000 items; allow generous slack.
    index = _bloom_over(list(range(1000)), pa.int64())
    absent = [v for v in range(10_000, 11_000) if not index.contains(v)]
    assert len(absent) > 900, "the index should still discriminate"


@pytest.mark.integration
def test_nulls_are_not_indexed_and_do_not_break_the_reader():
    index = _bloom_over([1, None, 3], pa.int64())
    assert index.contains(1)
    assert index.contains(3)


@pytest.mark.integration
def test_unindexable_column_types_produce_no_index():
    # A bloom must not exist for a type the reader cannot encode; otherwise a `date`
    # literal would probe an index built over some other byte encoding.
    for values, dtype in (
        ([1.5, 2.5], pa.float64()),
        ([True, False], pa.bool_()),
        ([1, 2], pa.date32()),
    ):
        batch = pa.record_batch([pa.array(values, dtype)], names=["c"])
        assert nat.build_column_bloom([batch], 0, 2) is None, f"{dtype} must not be indexed"


@pytest.mark.integration
def test_probing_across_type_domains_reports_a_false_absence():
    """The hazard the type-domain guard exists to stop.

    An Int64 index hashes 8 little-endian bytes; the string `"5"` hashes one byte. Probing
    one with the other reports a *definitive absence* for a value that is present — and
    `zonemap_prune_filter` turns a definitive absence into an empty relation.
    """
    ints = _bloom_over([1, 2, 3, 5, 8], pa.int64())
    assert ints.contains(5)
    assert not ints.contains("5")  # present value, cross-domain probe -> "absent"

    strings = _bloom_over(["1", "2", "5"], pa.utf8())
    assert strings.contains("5")
    assert not strings.contains(5)


@pytest.mark.integration
def test_join_key_bloom_matches_signed_zero_and_nan():
    """The distributed-join key bloom (`build_key_bloom` on the small side, probed by
    `bloom_filter_batches` on the large side to drop non-matching rows *before* the shuffle)
    must fold float keys to the engine's identity, exactly as the equi-join's own keys do.

    An equi-join matches `-0.0` to `0.0` and every NaN to one NaN. If the bloom is built on
    `0.0`'s raw bytes and probed on `-0.0`'s, it reports "absent" and drops a probe row the
    join *would* have matched — a silent distributed wrong answer. This pins the fold.
    """
    import struct

    def f64(bits: int) -> float:
        return struct.unpack("<d", struct.pack("<Q", bits))[0]

    neg_nan = f64(0xFFF8000000000000)
    pos_nan = f64(0x7FF8000000000000)
    build = pa.record_batch([pa.array([0.0, 1.5, pos_nan], pa.float64())], names=["k"])
    probe = pa.record_batch([pa.array([-0.0, neg_nan, 1.5, 9.9], pa.float64())], names=["k"])
    bloom = nat.build_key_bloom([build], [0], 8)
    kept = nat.bloom_filter_batches([probe], [0], bloom)
    survivors = pa.Table.from_batches(kept).column("k").to_pylist() if kept else []
    # -0.0 matches 0.0; neg_nan matches the pos NaN in the build side; 1.5 matches; 9.9 not.
    assert any(s == 0.0 for s in survivors), "-0.0 must survive (matches 0.0)"
    assert any(s != s for s in survivors), "a negative NaN must survive (matches NaN)"
    assert 1.5 in survivors
    assert 9.9 not in survivors, "a genuinely absent key must still be dropped"


@pytest.mark.integration
def test_the_pruning_rule_refuses_a_cross_domain_probe():
    from batcher.kyber.rules.zonemap_pruning import _same_bloom_domain
    from batcher.plan.stats import ColumnStat

    int_col = ColumnStat(min=1, max=8, bloom=b"")
    str_col = ColumnStat(min="a", max="h", bloom=b"")

    assert _same_bloom_domain(int_col, 5)
    assert not _same_bloom_domain(int_col, "5")
    assert _same_bloom_domain(str_col, "e")
    assert not _same_bloom_domain(str_col, 5)
    # Unindexable literals and unbounded columns are never pruned on.
    assert not _same_bloom_domain(int_col, 5.0)
    assert not _same_bloom_domain(int_col, True)
    assert not _same_bloom_domain(ColumnStat(bloom=b""), 5)
