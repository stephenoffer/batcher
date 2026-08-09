"""EXIF orientation — the rotation a photo corpus carries and a decoder ignores.

A camera does not rotate its sensor data. It records which way up it was held in the EXIF
``Orientation`` tag and stores the pixels as read, so a portrait phone photo is *stored*
landscape with a "rotate 90" note. Every viewer honours the note; so do ``cv2.imread`` and
anything built on ``PIL.ImageOps.exif_transpose``. The decoder behind the ``.image``
namespace does not.

That gap is invisible in every way a test usually looks. The decode succeeds, the tensor
is the right shape, the pixels are real photograph pixels — they are just a quarter turn
from what the rest of the pipeline sees. So these tests assert on *where a known pixel
ended up*, which is the only thing that can tell a rotated image from an upright one.

The fixtures are asymmetric on purpose: an image that is symmetric under the transform
being tested cannot detect it.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

# Codes 5-8 involve a quarter turn, so they transpose the stored dimensions; 1-4 do not.
_QUARTER_TURNS = (5, 6, 7, 8)


def _tagged_jpeg(orientation: int, width: int = 8, height: int = 4) -> bytes:
    """A JPEG whose top-left quadrant is red, the rest green, tagged with `orientation`.

    One distinctly-coloured quadrant identifies any of the eight transforms — a mirror and
    a rotation move it to different corners — and a quadrant survives compression where a
    one-pixel marker would not.

    Written at 4:4:4 (`subsampling=0`) and quality 100 on purpose. At the default 4:2:0 the
    two chroma planes are half resolution, and Rust's `zune-jpeg` and libjpeg interpolate
    them back differently, so a pixel-for-pixel comparison against PIL measures *the
    decoders* rather than the transform under test. Turning subsampling off removes that
    variable entirely and leaves the assertion about the thing it is named for.
    """
    from PIL import Image

    img = Image.new("RGB", (width, height), (0, 255, 0))
    for x in range(width // 2):
        for y in range(height // 2):
            img.putpixel((x, y), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100, subsampling=0, exif=exif.tobytes())
    return buf.getvalue()


def _pil_reference(data: bytes):
    """What every `exif_transpose`-based pipeline sees for the same bytes."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(io.BytesIO(data)))


def test_exif_orientation_reports_the_tag():
    """You cannot fix what you cannot see, and a rotated decode looks like a fine one."""
    pytest.importorskip("PIL")
    rows = [_tagged_jpeg(code) for code in range(1, 9)]
    ds = bt.from_pydict({"i": [*rows, None]})
    got = ds.select(o=col("i").image.exif_orientation()).to_pydict()["o"]
    assert got[:8] == list(range(1, 9))
    assert got[8] is None


def test_an_image_with_no_tag_reports_upright_not_null():
    """`1` means "already upright", which is the honest answer for an untagged image.

    Null would say "unknown" and push every caller into a coalesce; a PNG is not of
    unknown orientation, it is of no orientation, which is the same thing as upright.
    """
    from batcher.plan.expr_ir.image import _PNG_1X1

    ds = bt.from_pydict({"i": [_PNG_1X1]})
    assert ds.select(o=col("i").image.exif_orientation()).to_pydict()["o"] == [1]


@pytest.mark.parametrize("code", range(1, 9))
def test_auto_orient_agrees_with_pil_exif_transpose(code):
    """The reference is what the rest of the ecosystem does, not what we think is right.

    Comparing against `ImageOps.exif_transpose` pixel for pixel is the whole point: any
    of the eight transforms can be gotten subtly wrong (a mirror in place of a rotation
    reads as plausible), and only the reference can say which one was meant.
    """
    import numpy as np

    pytest.importorskip("PIL")
    from PIL import Image

    data = _tagged_jpeg(code)
    ds = bt.from_pydict({"i": [data]})
    out = ds.select(b=col("i").image.auto_orient()).to_pydict()["b"][0]
    ours = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.int16)
    theirs = np.asarray(_pil_reference(data).convert("RGB"), dtype=np.int16)

    assert ours.shape == theirs.shape, f"orientation {code} produced the wrong dimensions"
    # Still a tolerance, because JPEG is lossy at any quality and the two decoders round
    # the inverse DCT differently. A wrong transform moves a quadrant by 255, far outside.
    assert np.abs(ours - theirs).max() <= 8, f"orientation {code} put the pixels elsewhere"


@pytest.mark.parametrize("code", _QUARTER_TURNS)
def test_a_quarter_turn_transposes_the_decoded_dimensions(code):
    """The coarsest check, and the one a `width > height` filter depends on.

    Without orienting, a corpus of portrait phone photos reports landscape dimensions,
    so an aspect-ratio filter selects exactly the wrong half of it.
    """
    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_tagged_jpeg(code, width=8, height=4)]})
    stored = ds.select(d=col("i").image.decode()).to_pydict()["d"][0]
    upright = ds.select(d=col("i").image.auto_orient().image.decode()).to_pydict()["d"][0]

    assert (stored["width"], stored["height"]) == (8, 4)
    assert (upright["width"], upright["height"]) == (4, 8)


def test_orienting_is_idempotent():
    """The result is PNG, which carries no EXIF, so the rotation cannot be applied twice.

    Getting this wrong is the classic EXIF bug: apply the transform, keep the tag, and
    the next tool in the chain rotates it again.
    """
    import numpy as np

    pytest.importorskip("PIL")
    from PIL import Image

    ds = bt.from_pydict({"i": [_tagged_jpeg(6)]})
    once = col("i").image.auto_orient()
    twice = once.image.auto_orient()
    out = ds.select(a=once, b=twice).to_pydict()
    a = np.asarray(Image.open(io.BytesIO(out["a"][0])).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(io.BytesIO(out["b"][0])).convert("RGB"), dtype=np.int16)

    assert a.shape == b.shape
    assert np.abs(a - b).max() == 0


def test_an_already_upright_image_is_unchanged():
    """Orienting a corpus must not damage the images that did not need it."""
    import numpy as np

    pytest.importorskip("PIL")
    from PIL import Image

    data = _tagged_jpeg(1)
    ds = bt.from_pydict({"i": [data]})
    out = ds.select(b=col("i").image.auto_orient()).to_pydict()["b"][0]
    ours = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"), dtype=np.int16)
    theirs = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)

    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() <= 8


def test_nulls_and_garbage_stay_null():
    """The media convention: a bad row is null, it never fails the batch."""
    ds = bt.from_pydict({"i": [None, b"not an image"]})
    out = ds.select(b=col("i").image.auto_orient(), o=col("i").image.exif_orientation()).to_pydict()
    assert out["b"] == [None, None]
    assert out["o"] == [None, None]


def test_the_oriented_bytes_feed_the_tensor_path():
    """The composition this exists for: orient, then decode to a model-ready tensor."""
    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_tagged_jpeg(6)]})
    tensor = col("i").image.auto_orient().image.to_tensor(8, 8)
    assert len(ds.select(t=tensor).to_pydict()["t"][0]) == 8 * 8 * 3


def test_a_column_of_nothing_but_nulls_is_not_a_type_error():
    """Arrow types an all-null column as `Null`, not as `Binary` with the nulls set.

    That shape arrives constantly in a media pipeline — a download stage where every fetch
    failed, an outer join that matched nothing, a partition filtered empty upstream — and
    every `.image` and `.audio` op used to *fail the batch* on it with `expected a Binary
    argument, got Null`. A type error about a column the caller never typed, on the one
    input whose answer was never in doubt.

    Parametrized across the namespace rather than the two ops that surfaced it, because
    the bug was in the shared entry point and a test of one op would let it back in
    through another.
    """
    import batcher as bt

    ops = {
        "decode": lambda c: c.image.decode(),
        "to_tensor": lambda c: c.image.to_tensor(4, 4),
        "to_tensor_f32": lambda c: c.image.to_tensor_f32(4, 4),
        "to_grayscale": lambda c: c.image.to_grayscale(4, 4),
        "center_crop": lambda c: c.image.center_crop(4, 4),
        "resize": lambda c: c.image.resize(4, 4),
        "crop": lambda c: c.image.crop(0, 0, 2, 2),
        "encode": lambda c: c.image.encode("png"),
        "convert": lambda c: c.image.convert("L"),
        "dhash": lambda c: c.image.dhash(),
        "brightness": lambda c: c.image.brightness(),
        "sharpness": lambda c: c.image.sharpness(),
        "thumbnail": lambda c: c.image.thumbnail(8),
        "letterbox": lambda c: c.image.letterbox(4, 4),
        "auto_orient": lambda c: c.image.auto_orient(),
        "exif_orientation": lambda c: c.image.exif_orientation(),
        "audio.decode": lambda c: c.audio.decode(),
        "audio.to_waveform": lambda c: c.audio.to_waveform(),
    }
    ds = bt.from_pydict({"i": [None, None]})
    assert ds.collect().schema.field("i").type == __import__("pyarrow").null()

    for label, build in ops.items():
        out = ds.select(x=build(col("i"))).to_pydict()["x"]
        assert out == [None, None], f"{label} did not answer null for an all-null column"
