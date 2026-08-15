"""The image geometry, photometric and fingerprinting expressions, over real encoded images.

These cover the operations an augmentation policy and a deduplication pass are written
from. What is asserted is what a naive implementation gets wrong while still producing
plausible-looking output:

- a geometry op that promotes RGB to RGBA, inflating every row by a third and changing the
  column's type;
- a re-encode that ignores the requested container, so a "shrink the corpus" step writes
  PNGs that are larger than the JPEGs they came from;
- a `posterize` that requantizes instead of masking, so it disagrees with `PIL`;
- histogram ops that divide by an empty range on a flat image;
- a perceptual hash that does not survive the rescaling it exists to survive.

Absolute pixel values are pinned only where they are exact by construction. Everything else
is asserted as an ordering, because the alternative is a test that fails on a resampling
change while saying nothing about whether the operation still works.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.integration


def _encode(array, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, fmt)
    return buf.getvalue()


def _flat(rgb: tuple[int, int, int], size: int = 16) -> bytes:
    return _encode(np.tile(np.array(rgb, dtype="uint8"), (size, size, 1)))


def _corner_marked(width: int = 8, height: int = 4) -> bytes:
    """Black, with a white pixel at the top-left — so a turn has a witness."""
    array = np.zeros((height, width, 3), dtype="uint8")
    array[0, 0] = 255
    return _encode(array)


def _decoded(expr) -> np.ndarray:
    """Run a bytes-out expression over one image and decode the result to an array."""
    out = bt.from_pydict({"img": [_MARKED]}).select(o=expr).to_pydict()["o"][0]
    return np.array(Image.open(io.BytesIO(out)))


_MARKED = _corner_marked()


def _one(image: bytes, **exprs) -> dict:
    return bt.from_pydict({"img": [image]}).select(**exprs).to_pydict()


# --- geometry ----------------------------------------------------------------------


def test_a_quarter_turn_moves_a_known_pixel_to_a_known_place() -> None:
    """A transposed *shape* is not evidence: the pixel has to land in the right corner."""
    turned = _decoded(bt.col("img").image.rotate(90))
    assert turned.shape[:2] == (8, 4)
    assert tuple(turned[0, 3]) == (255, 255, 255)


def test_rotation_angles_are_normalized_rather_than_rejected() -> None:
    """`-90` and `270` name the same turn; a full turn is the identity."""
    assert np.array_equal(
        _decoded(bt.col("img").image.rotate(-90)), _decoded(bt.col("img").image.rotate(270))
    )
    assert np.array_equal(
        _decoded(bt.col("img").image.rotate(360)), np.array(Image.open(io.BytesIO(_MARKED)))
    )


def test_a_free_rotation_is_refused_at_plan_build() -> None:
    """Refused where it names the caller's method, not per row a million rows into a scan."""
    with pytest.raises(Exception, match="multiple of 90"):
        bt.col("img").image.rotate(45)


def test_flips_mirror_the_axis_they_name() -> None:
    flipped = _decoded(bt.col("img").image.flip_horizontal())
    assert tuple(flipped[0, 7]) == (255, 255, 255)
    assert tuple(flipped[0, 0]) == (0, 0, 0)
    vertical = _decoded(bt.col("img").image.flip_vertical())
    assert tuple(vertical[3, 0]) == (255, 255, 255)


def test_geometry_does_not_add_an_alpha_channel_to_an_rgb_image() -> None:
    """The helpers underneath hand back RGBA whatever went in.

    Left alone, a flipped corpus silently grows a fourth channel: a third more bytes per
    row and a changed column type, from an operation that only moved pixels.
    """
    for expr in (
        bt.col("img").image.flip_horizontal(),
        bt.col("img").image.rotate(180),
        bt.col("img").image.invert(),
        bt.col("img").image.blur(1.0),
    ):
        got = _one(_MARKED, d=expr.image.decode())["d"][0]
        assert got["channels"] == 3, f"{expr} promoted RGB to {got['mode']}"


def test_pad_centres_without_resampling_and_fills_the_rest() -> None:
    """The difference from `letterbox`: every surviving pixel keeps its exact value."""
    source = _flat((200, 100, 50), size=4)
    out = (
        bt.from_pydict({"img": [source]})
        .select(o=bt.col("img").image.pad(8, 8, fill=7))
        .to_pydict()["o"][0]
    )
    padded = np.array(Image.open(io.BytesIO(out)))
    assert padded.shape[:2] == (8, 8)
    assert tuple(padded[0, 0]) == (7, 7, 7)
    assert tuple(padded[3, 3]) == (200, 100, 50)


# --- containers --------------------------------------------------------------------


def test_every_bytes_out_op_can_write_the_container_it_is_asked_for() -> None:
    """`format` used to be `encode`'s alone, so a resize inflated a JPEG corpus into PNG."""
    source = _flat((30, 60, 90), size=32)
    got = _one(
        source,
        resized=bt.col("img").image.resize(16, 16, format="jpeg").image.format(),
        flipped=bt.col("img").image.flip_horizontal(format="bmp").image.format(),
        oriented=bt.col("img").image.auto_orient(format="gif").image.format(),
    )
    assert got == {"resized": ["jpeg"], "flipped": ["bmp"], "oriented": ["gif"]}


def test_jpeg_quality_trades_size_for_fidelity() -> None:
    """A quality knob that does not change the encoding is not a quality knob."""
    noisy = _encode(np.indices((64, 64)).sum(0) % 256)
    sizes = {
        q: len(
            bt.from_pydict({"img": [noisy]})
            .select(o=bt.col("img").image.resize(64, 64, format="jpeg", quality=q))
            .to_pydict()["o"][0]
        )
        for q in (20, 95)
    }
    assert sizes[20] < sizes[95]


def test_an_unwritable_container_is_refused_at_plan_build() -> None:
    with pytest.raises(Exception, match="format must be one of"):
        bt.col("img").image.resize(8, 8, format="webp")


# --- photometric -------------------------------------------------------------------


def test_a_factor_of_one_is_the_identity_for_every_enhancement() -> None:
    """The `PIL.ImageEnhance` convention an augmentation policy is written against."""
    source = _encode(np.indices((16, 16)).sum(0).astype("uint8")[:, :, None].repeat(3, 2))
    original = np.array(Image.open(io.BytesIO(source)))
    for expr in (
        bt.col("img").image.adjust_brightness(1.0),
        bt.col("img").image.adjust_contrast(1.0),
        bt.col("img").image.adjust_saturation(1.0),
    ):
        out = bt.from_pydict({"img": [source]}).select(o=expr).to_pydict()["o"][0]
        assert np.array_equal(np.array(Image.open(io.BytesIO(out))), original)


def test_brightness_clamps_at_white_rather_than_wrapping() -> None:
    """100 * 4 is 400, which must land on 255 rather than on 144."""
    got = _one(
        _flat((100, 100, 100)), c=bt.col("img").image.adjust_brightness(4.0).image.mean_color()
    )
    assert got["c"][0] == {"r": 255.0, "g": 255.0, "b": 255.0}


def test_zero_saturation_leaves_three_identical_channels() -> None:
    source = _flat((10, 200, 30))
    got = _one(source, g=bt.col("img").image.adjust_saturation(0.0).image.is_grayscale())
    assert got["g"] == [True]


def test_posterize_masks_the_low_bits_the_way_pillow_does() -> None:
    """A requantizing implementation disagrees with `PIL.ImageOps.posterize` at 1 bit."""
    got = _one(_flat((64, 255, 130)), c=bt.col("img").image.posterize(1).image.mean_color())
    assert got["c"][0] == {"r": 0.0, "g": 128.0, "b": 128.0}


def test_solarize_inverts_only_at_and_above_the_threshold() -> None:
    got = _one(_flat((10, 200, 255)), c=bt.col("img").image.solarize(128).image.mean_color())
    assert got["c"][0] == {"r": 10.0, "g": 55.0, "b": 0.0}


def test_a_flat_image_survives_the_histogram_operations() -> None:
    """Both divide by the range they find, and a solid tile has none."""
    source = _flat((128, 128, 128))
    for expr in (bt.col("img").image.equalize(), bt.col("img").image.autocontrast()):
        got = _one(source, e=expr.image.entropy())
        assert got["e"] == [0.0], "a solid tile came back with variation in it"


def test_autocontrast_stretches_a_narrow_range_to_the_full_one() -> None:
    ramp = _encode(
        np.tile(np.array([100, 110, 120, 130], dtype="uint8"), (4, 1))[:, :, None].repeat(3, 2)
    )
    out = (
        bt.from_pydict({"img": [ramp]})
        .select(o=bt.col("img").image.autocontrast())
        .to_pydict()["o"][0]
    )
    stretched = np.array(Image.open(io.BytesIO(out)))
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_blur_softens_and_sharpen_hardens_the_same_edge() -> None:
    edge = _encode(
        np.where(np.indices((64, 64))[1] < 32, 0, 255).astype("uint8")[:, :, None].repeat(3, 2)
    )
    detail = _one(
        edge,
        base=bt.col("img").image.sharpness(),
        soft=bt.col("img").image.blur(3.0).image.sharpness(),
        crisp=bt.col("img").image.sharpen(2.0).image.sharpness(),
    )
    assert detail["soft"][0] < detail["base"][0]
    assert detail["crisp"][0] >= detail["base"][0]


# --- fingerprints ------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ahash", "dhash", "phash"])
def test_a_hash_survives_rescaling_and_separates_different_pictures(method: str) -> None:
    """The two properties a perceptual hash exists for, in one test.

    Surviving rescaling is what makes it find the same picture at three resolutions;
    separating different pictures is what stops it matching everything.
    """
    scene = np.indices((128, 128)).sum(0).astype("uint8")[:, :, None].repeat(3, 2)
    scene[:, :, 1] = np.indices((128, 128))[0].astype("uint8")
    small = np.array(Image.fromarray(scene).resize((32, 32)))
    other = (
        np.where(np.indices((128, 128))[1] % 8 < 4, 0, 255).astype("uint8")[:, :, None].repeat(3, 2)
    )

    ds = bt.from_pydict({"img": [_encode(scene), _encode(small), _encode(other)]})
    digests = ds.select(h=getattr(bt.col("img").image, method)()).to_pydict()["h"]
    rescaled = bin((digests[0] ^ digests[1]) & 0xFFFFFFFFFFFFFFFF).count("1")
    different = bin((digests[0] ^ digests[2]) & 0xFFFFFFFFFFFFFFFF).count("1")
    assert rescaled <= 8, f"{method} moved {rescaled} bits under a 4x downscale"
    assert different > rescaled, f"{method} put two different pictures {different} bits apart"


def test_a_hash_is_expressible_as_a_similarity_predicate() -> None:
    """The whole point: the fingerprint has to be usable as ordinary integer arithmetic."""
    scene = _encode(np.indices((64, 64)).sum(0).astype("uint8")[:, :, None].repeat(3, 2))
    ds = bt.from_pydict({"a": [scene], "b": [scene]})
    near = ds.select(
        close=bt.col("a").image.phash().bitwise_xor(bt.col("b").image.phash()).bit_count() <= 6
    ).to_pydict()
    assert near == {"close": [True]}


# --- header-only facts -------------------------------------------------------------


def test_the_header_facts_come_from_the_bytes_not_the_path() -> None:
    """A corpus downloaded by content type is full of files whose extension lies."""
    wide = _encode(np.zeros((10, 40, 3), dtype="uint8"))
    as_jpeg = _encode(np.zeros((10, 40, 3), dtype="uint8"), fmt="JPEG")
    got = (
        bt.from_pydict({"img": [wide, as_jpeg]})
        .select(
            ratio=bt.col("img").image.aspect_ratio(),
            alpha=bt.col("img").image.has_alpha(),
            container=bt.col("img").image.format(),
        )
        .to_pydict()
    )
    assert got["ratio"] == [4.0, 4.0]
    assert got["alpha"] == [False, False]
    assert got["container"] == ["png", "jpeg"]


def test_an_rgba_image_reports_its_alpha_channel() -> None:
    rgba = _encode(np.zeros((4, 4, 4), dtype="uint8"))
    got = _one(rgba, a=bt.col("img").image.has_alpha())
    assert got["a"] == [True]


def test_every_new_operation_answers_null_for_a_null_or_undecodable_row() -> None:
    """The namespace's convention: a corrupt row is a null, never a failed batch."""
    ds = bt.from_pydict({"img": [None, b"not an image"]})
    got = ds.select(
        turned=bt.col("img").image.rotate(90),
        bright=bt.col("img").image.adjust_brightness(2.0),
        digest=bt.col("img").image.phash(),
        colour=bt.col("img").image.colorfulness(),
        container=bt.col("img").image.format(),
    ).to_pydict()
    for name, values in got.items():
        assert values == [None, None], f"{name} did not null an unreadable row"


def test_streaming_a_transform_pipeline_matches_collecting_it() -> None:
    """The new ops must mean the same thing on every path the engine runs them on.

    `collect` and `iter_batches` are different executors, and a per-row kernel that reads
    anything batch-scoped — a histogram, a parameter resolved once — is exactly where the
    two drift apart. Asserted positionally, because the order is part of the answer.
    """
    corpus = [_flat((i * 8, 255 - i * 8, 40), size=16) for i in range(24)]
    ds = bt.from_pydict({"img": corpus})
    query = ds.select(
        digest=bt.col("img").image.phash(),
        colour=bt.col("img").image.colorfulness(),
        even=bt.col("img").image.equalize().image.entropy(),
        turned=bt.col("img").image.rotate(90, format="jpeg").image.format(),
    )
    collected = query.collect().to_pydict()
    streamed: dict[str, list] = {name: [] for name in collected}
    for batch in query.iter_batches(batch_size=5):
        for name, values in batch.to_pydict().items():
            streamed[name] += values
    assert streamed == collected
