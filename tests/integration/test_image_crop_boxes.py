"""Cropping a bounding box that varies per row — the detection pipeline's core operation.

`.image.crop` took four constants, so the one thing it could not express was the thing a
vision pipeline is built around: cut the box a detector predicted out of the frame it was
predicted in. Boxes are data, one per row, so a literal-only window left a per-row Python
loop as the only way to do it.

Every fixture here is an image whose four quadrants are four distinct colours, so a crop
lands on a colour that names which window was actually used. That is the only kind of
assertion that separates "each row got its own window" from "every row got row zero's" —
a kernel that read its bounds once for the batch returns four identically-sized, perfectly
valid patches and passes any test that only checks shapes.
"""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

#: The four quadrant colours, in the order `(left-top, right-top, left-bottom, right-bottom)`.
_QUADRANTS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def _quadrant_png(size: int = 8) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (size, size))
    pixels = img.load()
    half = size // 2
    for x in range(size):
        for y in range(size):
            pixels[x, y] = _QUADRANTS[(x >= half) + 2 * (y >= half)]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _corner(png: bytes) -> tuple[int, int, int]:
    """The top-left pixel of a crop, which names the quadrant it came from."""
    import numpy as np
    from PIL import Image

    return tuple(int(v) for v in np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))[0, 0])


@pytest.fixture
def boxes():
    """One frame, four rows, one box per quadrant."""
    pytest.importorskip("PIL")
    frame = _quadrant_png()
    return bt.from_pydict(
        {
            "img": [frame] * 4,
            "bx": [0, 4, 0, 4],
            "by": [0, 0, 4, 4],
            "bw": [4, 4, 4, 4],
            "bh": [4, 4, 4, 4],
        }
    )


def test_each_row_is_cropped_by_its_own_box(boxes):
    """The whole point: four rows, four windows, four different patches."""
    patches = boxes.select(
        p=col("img").image.crop(col("bx"), col("by"), col("bw"), col("bh"))
    ).to_pydict()["p"]

    assert [_corner(p) for p in patches] == _QUADRANTS


def test_constant_bounds_still_work(boxes):
    """The literal form is the same operation with constant columns, not a second one."""
    patches = boxes.select(p=col("img").image.crop(4, 4, 4, 4)).to_pydict()["p"]

    assert {_corner(p) for p in patches} == {_QUADRANTS[3]}


def test_constants_and_columns_mix(boxes):
    """A fixed-size patch at a per-row position is the common half-and-half case."""
    patches = boxes.select(p=col("img").image.crop(col("bx"), col("by"), 4, 4)).to_pydict()["p"]

    assert [_corner(p) for p in patches] == _QUADRANTS


@pytest.mark.parametrize(
    ("label", "bx", "by", "bw", "bh"),
    [
        ("negative offset", -1, 0, 4, 4),
        ("zero extent", 0, 0, 0, 4),
        ("negative extent", 0, 0, -4, 4),
        ("null bound", 0, 0, 4, None),
        ("window past the image", 99, 0, 4, 4),
    ],
)
def test_an_unusable_window_nulls_only_its_own_row(label, bx, by, bw, bh):
    """A box the caller could not supply is a row with no answer, not a failed batch.

    This is the semantic that had to change when the bounds became data. As constants a
    negative offset was a *query* error and rightly raised; per row it is one bad box among
    thousands — from a detector that declined to predict, or a join that matched nothing —
    and failing the batch for it would take every good row down with it.
    """
    pytest.importorskip("PIL")
    frame = _quadrant_png()
    ds = bt.from_pydict(
        {
            "img": [frame, frame, frame],
            "bx": [0, bx, 0],
            "by": [0, by, 0],
            "bw": [4, bw, 4],
            "bh": [4, bh, 4],
        }
    )
    out = ds.select(
        p=col("img").image.crop(col("bx"), col("by"), col("bw"), col("bh"))
    ).to_pydict()["p"]

    assert out[1] is None, label
    assert out[0] is not None and out[2] is not None, f"{label} took its neighbours with it"


def test_a_window_running_past_an_edge_is_clipped_not_padded(boxes):
    """The deliberate difference from `center_crop`: a crop is looked at, so inventing
    black pixels there would be inventing data."""
    pytest.importorskip("PIL")
    from PIL import Image

    ds = bt.from_pydict({"img": [_quadrant_png()], "bx": [6], "by": [6], "bw": [100], "bh": [100]})
    out = ds.select(
        p=col("img").image.crop(col("bx"), col("by"), col("bw"), col("bh"))
    ).to_pydict()["p"]

    assert Image.open(io.BytesIO(out[0])).size == (2, 2)


def test_a_crop_is_an_image_column_the_rest_of_the_namespace_reads(boxes):
    """The composition a detection pipeline writes: crop the box, then feed a model."""
    patch = col("img").image.crop(col("bx"), col("by"), col("bw"), col("bh"))
    out = boxes.select(
        dims=patch.image.decode(),
        tensor=patch.image.letterbox(8, 8),
    ).to_pydict()

    assert all(d == {"width": 4, "height": 4, "channels": 4, "mode": "RGBA"} for d in out["dims"])
    assert all(len(t) == 8 * 8 * 3 for t in out["tensor"])


def test_the_result_is_priced_as_encoded_bytes():
    """Rows genuinely differ in size, so it cannot be a fixed-shape tensor — and the
    planner must not size it as one."""
    import pyarrow as pa

    from batcher.plan.schema import SchemaRef
    from batcher.plan.types import infer_type

    schema = SchemaRef.from_arrow(pa.schema([("img", pa.binary()), ("bx", pa.int64())]))
    expr = col("img").image.crop(col("bx"), 0, 4, 4)

    assert infer_type(expr, schema) == pa.binary()


def test_streaming_agrees_with_collecting(boxes):
    """Bounds are read per row, so batching must not change which window a row got."""
    expr = col("img").image.crop(col("bx"), col("by"), col("bw"), col("bh"))
    whole = boxes.select(p=expr).to_pydict()["p"]
    streamed = [
        row
        for batch in boxes.select(p=expr).iter_batches(batch_size=1)
        for row in batch.column("p").to_pylist()
    ]

    assert whole == streamed
