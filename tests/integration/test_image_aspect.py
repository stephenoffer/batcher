"""Aspect-preserving resizes — `.image.thumbnail` and `.image.letterbox`.

`resize` and `to_tensor` both take two dimensions and stretch whatever does not already
match them. That is right for a fixed-input model fed square crops and wrong for almost
everything else, and the failure is silent in the way that matters: a squashed image has
exactly the shape that was asked for, so no assertion on dimensions can see it.

So the tests here assert on *geometry* — that a circle stays round, that padding lands on
the sides it should — which is the only thing that separates these two operations from the
ones they exist beside.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration

#: The `letterbox` default fill, YOLO's grey. Named so a test reads as intent.
_YOLO_GREY = 114


def _png(width: int, height: int, colour=(200, 40, 40)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


def _dims(data: bytes) -> tuple[int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(data)).size


def _thumb(source: bytes, max_size: int) -> bytes:
    ds = bt.from_pydict({"i": [source]})
    return ds.select(t=col("i").image.thumbnail(max_size)).to_pydict()["t"][0]


@pytest.mark.parametrize(
    ("size", "max_size", "expected"),
    [
        ((400, 200), 100, (100, 50)),  # landscape: width binds
        ((200, 400), 100, (50, 100)),  # portrait: height binds
        ((300, 300), 100, (100, 100)),  # square
        ((1000, 3), 100, (100, 1)),  # a panorama's short side floors at one pixel
    ],
)
def test_thumbnail_scales_the_longest_side_and_keeps_the_ratio(size, max_size, expected):
    """The longest side hits `max_size` and the other follows the source's ratio.

    The panorama case is the one that bites: a truncating scale takes its 3-pixel side to
    zero, and a zero-dimension resize is an error, so the row would drop out entirely
    rather than thumbnail.
    """
    pytest.importorskip("PIL")
    assert _dims(_thumb(_png(*size), max_size)) == expected


def test_thumbnail_never_upscales():
    """Enlarging to reach `max_size` invents detail and costs bytes.

    It is also what `PIL.Image.thumbnail` does, so a corpus already normalized against
    that stays comparable rather than quietly growing.
    """
    pytest.importorskip("PIL")
    assert _dims(_thumb(_png(20, 10), 500)) == (20, 10)


def test_thumbnail_keeps_the_ratio_where_resize_does_not():
    """The distinction the operation exists for, stated as a comparison.

    A test that only checked `thumbnail` produced a valid PNG would pass just as well
    against `resize`, which is the operation being distinguished from.
    """
    pytest.importorskip("PIL")
    source = _png(400, 100)
    ds = bt.from_pydict({"i": [source]})
    out = ds.select(
        t=col("i").image.thumbnail(200),
        r=col("i").image.resize(200, 200),
    ).to_pydict()
    assert _dims(out["t"][0]) == (200, 50), "thumbnail must preserve the 4:1 ratio"
    assert _dims(out["r"][0]) == (200, 200), "resize is the one that stretches"


def test_letterbox_pads_rather_than_stretching_or_cropping():
    """A wide image on a square canvas keeps its shape; the bars go top and bottom.

    Asserting *where* the padding lands is the point. A stretch produces the same byte
    count, and so does a crop; only the position of the fill separates the three.
    """
    import numpy as np

    pytest.importorskip("PIL")
    # 40x10 red on a 20x20 canvas: scaled to 20x5, so 7 grey rows above and 8 below.
    ds = bt.from_pydict({"i": [_png(40, 10)]})
    flat = ds.select(t=col("i").image.letterbox(20, 20)).to_pydict()["t"][0]
    canvas = np.asarray(flat, dtype=np.uint8).reshape(20, 20, 3)

    assert (canvas[0] == _YOLO_GREY).all(), "the top row should be padding"
    assert (canvas[-1] == _YOLO_GREY).all(), "the bottom row should be padding"
    middle = canvas[10]
    assert middle[:, 0].min() > 150, "the middle row should be the image, not padding"
    # Every column carries image, because the width was the binding dimension.
    assert (canvas[:, 0] != _YOLO_GREY).any()


def test_letterbox_centres_the_image():
    """Off-centre padding biases every coordinate a detection model predicts."""
    import numpy as np

    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_png(40, 10)]})
    flat = ds.select(t=col("i").image.letterbox(20, 20)).to_pydict()["t"][0]
    canvas = np.asarray(flat, dtype=np.uint8).reshape(20, 20, 3)

    image_rows = [y for y in range(20) if (canvas[y] != _YOLO_GREY).any()]
    above, below = image_rows[0], 19 - image_rows[-1]
    assert abs(above - below) <= 1, f"padding split {above}/{below}, not centred"


def test_letterbox_fill_is_configurable():
    """A model trained against a different padding colour needs to say so."""
    import numpy as np

    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_png(40, 10)]})
    flat = ds.select(t=col("i").image.letterbox(20, 20, fill=0)).to_pydict()["t"][0]
    canvas = np.asarray(flat, dtype=np.uint8).reshape(20, 20, 3)
    assert (canvas[0] == 0).all()


def test_letterbox_is_a_fixed_shape_tensor_column():
    """It feeds a model, so the shape has to travel with the data like `to_tensor`'s."""
    import pyarrow as pa

    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_png(40, 10), None]})
    table = ds.select(t=col("i").image.letterbox(8, 6)).collect()
    field = table.schema.field("t")
    assert isinstance(field.type, pa.FixedShapeTensorType)
    assert tuple(field.type.shape) == (6, 8, 3)
    assert table.column("t").to_pylist()[1] is None


def test_the_planner_prices_a_letterbox_as_pixels_and_a_thumbnail_as_bytes():
    """One produces a tensor and one produces a compressed still; sizing must not confuse
    them, or a morsel is cut for a column that did not need it."""
    import pyarrow as pa

    from batcher.plan.types.media import imagefunc_type

    assert imagefunc_type(col("i").image.letterbox(640, 480)) == pa.fixed_shape_tensor(
        pa.uint8(), (480, 640, 3)
    )
    assert imagefunc_type(col("i").image.thumbnail(256)) == pa.binary()


def test_nulls_and_garbage_stay_null():
    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [None, b"not an image"]})
    out = ds.select(t=col("i").image.thumbnail(64), b=col("i").image.letterbox(8, 8)).to_pydict()
    assert out["t"] == [None, None]
    assert out["b"] == [None, None]


@pytest.mark.parametrize("bad", [-1, 256, 999])
def test_an_out_of_range_fill_is_rejected_at_plan_build(bad):
    """A byte that is not a byte is a caller bug, so it fails before any decode runs."""
    with pytest.raises(PlanError):
        col("i").image.letterbox(8, 8, fill=bad)


def test_the_detection_preprocessing_this_exists_for():
    """The composition a detection pipeline actually writes, end to end."""
    pytest.importorskip("PIL")
    ds = bt.from_pydict({"i": [_png(1280, 720), _png(720, 1280)]})
    boxed = ds.select(x=col("i").image.auto_orient().image.letterbox(64, 64)).to_pydict()["x"]
    # Both orientations land on the same canvas, which is what makes them batchable.
    assert [len(row) for row in boxed] == [64 * 64 * 3, 64 * 64 * 3]
