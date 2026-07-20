"""`.image.dhash()` — the primitive that makes image near-duplicate detection expressible.

Text near-dup detection has existed (`.str.minhash`, `.list.simhash`, `ds.ml.near_duplicates`),
but there was no image-level equivalent, so the largest single job in multimodal curation —
dropping the same picture re-encoded, rescaled or re-cropped — had no primitive at all.

These tests exercise the *pipeline*, not just the kernel: the kernel's own perceptual
properties are pinned in Rust (`bc-expr/src/eval/media/image.rs`). What matters here is
that the hash composes with the existing relational and bitwise vocabulary — no new
operator was needed for dedup, which is why this is a small addition rather than a
subsystem.
"""

from __future__ import annotations

import io
import math

import pytest

import batcher as bt

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.unit


def _wave_png(size: int, cycles: float) -> bytes:
    """A smooth 2-D wave — a gradient-rich image whose hash is not degenerate.

    A flat or monotonic image hashes to 0 (every gradient comparison is false), which
    would make any equality assertion below vacuously true.
    """
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            v = int(
                128
                + 120
                * math.sin(x / size * cycles * 2 * math.pi)
                * math.cos(y / size * 2 * math.pi)
            )
            v = max(0, min(255, v))
            px[x, y] = (v, v, v)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _hashes(images: list[bytes | None]) -> list[int | None]:
    ds = bt.from_pydict({"img": images})
    return ds.select(h=bt.col("img").image.dhash()).to_pydict()["h"]


def test_identical_images_share_a_hash() -> None:
    png = _wave_png(64, 3)
    got = _hashes([png, png])

    assert got[0] == got[1]
    assert got[0] not in (0, None), "degenerate hash would make this vacuous"


def test_a_rescaled_copy_is_within_the_near_duplicate_threshold() -> None:
    """The point of a *perceptual* hash: a thumbnail matches its original."""
    original, thumbnail = _hashes([_wave_png(256, 3), _wave_png(64, 3)])
    distance = bin(original ^ thumbnail).count("1")

    assert distance <= 8, f"a rescaled copy moved {distance} bits"


def test_a_different_image_is_far_away() -> None:
    a, b = _hashes([_wave_png(64, 3), _wave_png(64, 7)])
    distance = bin(a ^ b).count("1")

    assert distance >= 16, f"different images only {distance} bits apart"


def test_null_and_undecodable_rows_yield_null() -> None:
    """A corrupt image must not fail the batch — the whole corpus would be lost."""
    got = _hashes([_wave_png(32, 3), None, b"not an image at all"])

    assert got[0] is not None
    assert got[1] is None
    assert got[2] is None


def test_exact_duplicate_collapse_is_a_group_by() -> None:
    """No new operator: dedup by hash is the relational algebra that already exists."""
    a, b = _wave_png(64, 3), _wave_png(64, 7)
    ds = bt.from_pydict({"id": [1, 2, 3, 4], "img": [a, b, a, a]})

    deduped = ds.with_columns(h=bt.col("img").image.dhash()).distinct(subset=["h"])

    assert deduped.count() == 2


def test_near_duplicate_detection_is_a_join_plus_a_bit_count() -> None:
    """`bitwise_xor().bit_count()` is the Hamming distance — the existing vocabulary."""
    original, thumbnail, different = (
        _wave_png(256, 3),
        _wave_png(64, 3),
        _wave_png(64, 7),
    )
    left = bt.from_pydict({"lid": [1], "img": [original]}).with_columns(
        lh=bt.col("img").image.dhash()
    )
    right = bt.from_pydict({"rid": [2, 3], "img": [thumbnail, different]}).with_columns(
        rh=bt.col("img").image.dhash()
    )

    pairs = (
        left.select("lid", "lh")
        .cross_join(right.select("rid", "rh"))
        .with_columns(dist=bt.col("lh").bitwise_xor(bt.col("rh")).bit_count())
        .filter(bt.col("dist") <= 8)
    )

    # The rescaled copy matches; the different image does not.
    assert pairs.to_pydict()["rid"] == [2]
