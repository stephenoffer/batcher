"""Image brightness and sharpness, over real encoded images.

These are corpus-curation measures, so the tests are about the rows they exist to find: the
blank tile that decodes perfectly, the blown-out scan, and the out-of-focus photograph. Each
one passes every other check in an ingest pipeline.

Absolute sharpness values are deliberately not pinned. The measure is a normalized Laplacian
variance whose scale depends on the downsample filter, and a test asserting a remembered number
would fail on a resampling change while telling you nothing about whether blur is still
detectable. The orderings are what matter, so the orderings are what is asserted.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.integration


def _png(array) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def _flat(value: int, size: int = 64) -> bytes:
    return _png(np.full((size, size), value, dtype="uint8"))


def _checker(size: int = 64, block: int = 2) -> bytes:
    grid = np.indices((size, size)).sum(0) // block % 2
    return _png(grid * 255)


def _gradient(size: int = 64) -> bytes:
    return _png(np.tile(np.linspace(0, 255, size), (size, 1)))


def _measure(images: list, **exprs) -> dict:
    return bt.from_pydict({"img": images}).select(**exprs).to_pydict()


# --- brightness --------------------------------------------------------------------


def test_brightness_spans_black_to_white():
    got = _measure([_flat(0), _flat(128), _flat(255)], b=bt.col("img").image.brightness())["b"]
    assert got[0] == pytest.approx(0.0, abs=0.01)
    assert got[1] == pytest.approx(0.5, abs=0.02)
    assert got[2] == pytest.approx(1.0, abs=0.01)


def test_a_photograph_lands_between_the_extremes():
    """The separation the blank-tile filter relies on."""
    got = _measure([_flat(0), _checker(), _flat(255)], b=bt.col("img").image.brightness())["b"]
    assert got[0] < got[1] < got[2]


def test_the_blank_ends_of_a_corpus_can_be_filtered_out():
    images = [_flat(0), _checker(), _flat(255)]
    brightness = bt.col("img").image.brightness()
    kept = (
        bt.from_pydict({"img": images, "id": [1, 2, 3]})
        .filter((brightness > bt.lit(0.05)) & (brightness < bt.lit(0.95)))
        .to_pydict()
    )
    assert kept["id"] == [2]


def test_brightness_does_not_depend_on_the_source_resolution():
    """Downsampling first is what makes the cost independent of image size."""
    got = _measure([_flat(128, size=32), _flat(128, size=256)], b=bt.col("img").image.brightness())[
        "b"
    ]
    assert got[0] == pytest.approx(got[1], abs=0.02)


# --- sharpness ---------------------------------------------------------------------


def test_a_flat_image_has_no_sharpness():
    got = _measure([_flat(128)], s=bt.col("img").image.sharpness())["s"]
    assert got[0] == pytest.approx(0.0, abs=1e-6)


def test_detail_outscores_a_smooth_gradient_which_outscores_a_flat_field():
    """The ordering a focus filter depends on: detail > smooth variation > nothing."""
    got = _measure([_checker(), _gradient(), _flat(128)], s=bt.col("img").image.sharpness())["s"]
    detail, smooth, flat = got
    assert detail > smooth >= flat


def test_the_blurred_tail_of_a_corpus_can_be_filtered_out():
    images = [_checker(), _flat(128)]
    sharp = bt.col("img").image.sharpness()
    kept = bt.from_pydict({"img": images, "id": [1, 2]}).filter(sharp > bt.lit(1e-4)).to_pydict()
    assert kept["id"] == [1]


def test_sharpness_falls_as_an_image_is_blurred():
    """The property the measure is named for, against a progressively smoothed image."""
    base = np.indices((64, 64)).sum(0) // 2 % 2 * 255
    images = [_png(base)]
    for radius in (1, 3):
        blurred = Image.fromarray(base.astype("uint8")).filter(
            pytest.importorskip("PIL.ImageFilter").GaussianBlur(radius)
        )
        buf = io.BytesIO()
        blurred.save(buf, "PNG")
        images.append(buf.getvalue())
    got = _measure(images, s=bt.col("img").image.sharpness())["s"]
    assert got[0] > got[1] > got[2]


# --- shared contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [lambda e: e.image.brightness(), lambda e: e.image.sharpness()],
)
def test_both_measures_stay_inside_the_unit_interval(build):
    images = [_flat(0), _flat(255), _checker(), _gradient()]
    got = bt.from_pydict({"img": images}).select(v=build(bt.col("img"))).to_pydict()["v"]
    assert all(0.0 <= v <= 1.0 for v in got)


@pytest.mark.parametrize(
    "build",
    [lambda e: e.image.brightness(), lambda e: e.image.sharpness()],
)
def test_null_and_undecodable_input_yields_null_rather_than_failing_the_batch(build):
    images = [_checker(), None, b"this is not an image"]
    got = bt.from_pydict({"img": images}).select(v=build(bt.col("img"))).to_pydict()["v"]
    assert got[0] is not None
    assert got[1] is None
    assert got[2] is None


def test_a_tiny_image_is_not_enlarged_to_invent_an_interior():
    """`resize` fits within the box in both directions, so it would upscale a thumbnail —
    reporting a sharpness of 0 for an image with no interior pixel to measure. Only
    downscaling keeps the documented contract honest."""
    one_pixel = _png(np.full((1, 1), 200, dtype="uint8"))
    got = _measure(
        [one_pixel],
        s=bt.col("img").image.sharpness(),
        b=bt.col("img").image.brightness(),
    )
    assert got["s"] == [None]
    assert got["b"][0] == pytest.approx(200 / 255, abs=0.01)


def test_an_image_smaller_than_the_measure_box_still_scores():
    """Below the box but above the 3x3 floor: measured as-is, not resampled."""
    small = _checker(size=16)
    got = _measure([small], s=bt.col("img").image.sharpness())
    assert got["s"][0] > 0.0
