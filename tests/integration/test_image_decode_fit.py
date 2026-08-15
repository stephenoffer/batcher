"""How `read.images(decode=True)` reaches the shape it was asked for.

Every mode produces the same fixed-shape `(H, W, 3)` column, so no assertion on the
tensor's *shape* can tell them apart — which is exactly why the choice needs to be
explicit. `stretch` squashes an image whose aspect ratio is not the target's, and that is
a silent distortion: the decode succeeds, the tensor is the right size, and a detector
trained on letterboxed frames quietly predicts every box in the wrong place.

The three are separated here by what they put at a *known pixel*: padding fill, native
content, or resampled content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import batcher as bt

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.integration

#: The YOLO family's grey, which `letterbox` fills its leftover canvas with.
_LETTERBOX_FILL = 114


@pytest.fixture
def wide_image(tmp_path: Path) -> Path:
    """A 64x16 horizontal ramp — four times wider than tall, so `fit` has to matter."""
    ramp = np.tile(np.arange(64, dtype="uint8")[None, :, None], (16, 1, 3))
    Image.fromarray(ramp).save(tmp_path / "wide.png")
    return tmp_path


def _tensor(path: Path, fit: str, side: int = 32) -> np.ndarray:
    ds = bt.read.images(str(path), size=(side, side), fit=fit)
    return np.asarray(ds.select("image").collect().to_pydict()["image"][0]).reshape(side, side, 3)


def test_every_mode_produces_the_same_fixed_shape(wide_image: Path) -> None:
    """The reason the choice is invisible downstream, stated first."""
    for fit in ("stretch", "letterbox", "center_crop"):
        assert _tensor(wide_image, fit).shape == (32, 32, 3)


def test_stretch_fills_the_frame_with_image(wide_image: Path) -> None:
    """The default: every pixel of the frame is resampled content, distortion included."""
    corner = _tensor(wide_image, "stretch")[0, 0]
    assert corner.tolist() != [_LETTERBOX_FILL] * 3, "stretch must not pad"
    assert corner.max() < 10, "the ramp starts dark, so the top-left is near black"


def test_letterbox_pads_rather_than_distorting(wide_image: Path) -> None:
    """The whole image survives at its true ratio; the remainder is a constant.

    That constant is what makes it the object-detection choice: a model learns to ignore
    it, where a stretch moves every predicted box off its object.
    """
    tensor = _tensor(wide_image, "letterbox")
    assert tensor[0, 0].tolist() == [_LETTERBOX_FILL] * 3, "the top band should be fill"
    # The bands are told apart by variation, not by level: the source is a dark ramp whose
    # values run below the fill, so "brighter than the padding" would be the wrong test.
    top = tensor[0, :, 0]
    middle = tensor[16, :, 0]
    assert top.min() == top.max() == _LETTERBOX_FILL, "the pad band must be constant"
    assert middle.min() != middle.max(), "the middle band should carry the ramp"
    # A 4:1 image in a square box occupies a quarter of the height, centred.
    image_rows = [row for row in range(32) if tensor[row, :, 0].min() != tensor[row, :, 0].max()]
    assert 6 <= len(image_rows) <= 10, image_rows
    assert min(image_rows) >= 10 and max(image_rows) <= 21, image_rows


def test_center_crop_keeps_native_pixels(wide_image: Path) -> None:
    """No resampling: the centre pixel is a value the source actually had.

    The ramp's own values are its column indices, so a centre pixel that is not an
    original column index would mean the crop resampled.
    """
    centre = int(_tensor(wide_image, "center_crop")[8, 16, 0])
    assert centre in range(64), centre
    # The 64-wide ramp cropped to 32 columns starts at column 16, so column 16 of the
    # crop is source column 32 — exact, not interpolated.
    assert centre == 32


def test_an_unknown_fit_is_refused_at_plan_build(wide_image: Path) -> None:
    """Named at the call site, not as a silent fallback to the distorting default."""
    with pytest.raises(Exception, match="stretch"):
        bt.read.images(str(wide_image), size=(8, 8), fit="squish")


def test_the_default_is_unchanged(wide_image: Path) -> None:
    """Adding the option must not move what an existing pipeline gets."""
    explicit = _tensor(wide_image, "stretch")
    ds = bt.read.images(str(wide_image), decode=True, size=(32, 32))
    default = np.asarray(ds.select("image").collect().to_pydict()["image"][0]).reshape(32, 32, 3)
    assert np.array_equal(default, explicit)
