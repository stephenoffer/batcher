"""End-to-end image-decode expressions (`.image`) through the engine.

Exercises the full cross-layer path: the Python `.image` accessor → JSON IR
(``{"e":"image","fn":...}``) → Rust `Expr::Image` interpreter eval → Arrow result.
The JIT cannot compile library-backed decode, so this runs on the interpreter
oracle (the JIT cleanly falls back) — no separate assertion needed.
"""

from __future__ import annotations

import struct
import zlib

import pyarrow as pa
import pytest

import batcher as bt


def _png(width: int, height: int, rgb: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """A minimal solid-color RGB PNG, encoded with the stdlib (no image library)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * width  # filter byte 0 + pixels
    idat = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


_PNG_1x1 = _png(1, 1)


def test_image_decode_reads_dimensions():
    ds = bt.from_arrow(pa.table({"img": pa.array([_PNG_1x1], type=pa.binary())}))
    out = ds.select(dims=bt.col("img").image.decode()).collect()
    dims = out.column("dims")[0].as_py()
    assert dims == {"width": 1, "height": 1}


def test_image_to_tensor_shape():
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    out = ds.select(t=bt.col("img").image.to_tensor(2, 2)).collect()
    tensor = out.column("t")[0].as_py()
    assert len(tensor) == 2 * 2 * 3  # H*W*3, RGB8
    assert tensor[:3] == [255, 0, 0]  # solid red survives the resize


def test_image_to_tensor_f32_scales_and_normalizes():
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    # Bare: scale to [0, 1]. Solid red → first pixel (1.0, 0.0, 0.0) in HWC.
    out = ds.select(t=bt.col("img").image.to_tensor_f32(2, 2)).collect()
    tensor = out.column("t")[0].as_py()
    assert len(tensor) == 2 * 2 * 3
    assert tensor[:3] == pytest.approx([1.0, 0.0, 0.0])

    # Normalized + channels-first → the CHW red plane is (1 - 0.5)/0.25 = 2.0.
    t = bt.col("img").image.to_tensor_f32(
        2, 2, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25], channels_first=True
    )
    out = ds.select(t=t).collect()
    tensor = out.column("t")[0].as_py()
    assert tensor[0] == pytest.approx(2.0)  # red plane
    assert tensor[4] == pytest.approx(-2.0)  # green plane starts at index hw=4


def test_image_to_tensor_f32_is_a_shaped_float_tensor():
    # The output round-trips as a fixed-shape-tensor column (element type float32),
    # so it feeds a model with no per-batch Python re-type.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    out = ds.select(t=bt.col("img").image.to_tensor_f32(2, 2, channels_first=True)).collect()
    ttype = str(out.schema.field("t").type)
    assert ttype.startswith("extension<arrow.fixed_shape_tensor")
    assert "float" in ttype and "double" not in ttype
    # CHW shape travels in the metadata.
    assert "shape=[3,2,2]" in ttype


def test_image_center_crop_takes_the_middle():
    # A solid image crops to a solid tensor of the requested shape; the Rust unit test
    # (center_crop_takes_the_middle_and_pads_when_smaller) pins the actual pixel placement.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(8, 8), None], type=pa.binary())}))
    out = ds.select(t=bt.col("img").image.center_crop(4, 4)).collect()
    tensor = out.column("t")[0].as_py()
    assert len(tensor) == 4 * 4 * 3
    assert tensor[:3] == [255, 0, 0]  # solid red survives the crop
    assert out.column("t")[1].as_py() is None  # undecodable → null
    # The output is a shaped fixed-shape-tensor column.
    ttype = str(out.schema.field("t").type)
    assert ttype.startswith("extension<arrow.fixed_shape_tensor")
    assert "shape=[4,4,3]" in ttype


def test_image_center_crop_pads_when_smaller():
    # Cropping a 2x2 image to 4x4 zero-pads the border (torchvision CenterCrop).
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(2, 2)], type=pa.binary())}))
    tensor = ds.select(t=bt.col("img").image.center_crop(4, 4)).collect().column("t")[0].as_py()
    assert len(tensor) == 4 * 4 * 3
    assert tensor[:3] == [0, 0, 0]  # top-left corner is padding


def test_image_to_grayscale_single_channel():
    # Solid red → luma = round(0.299 * 255) = 76; output is one byte per pixel.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(8, 8), None], type=pa.binary())}))
    out = ds.select(g=bt.col("img").image.to_grayscale(4, 4)).collect()
    gray = out.column("g")[0].as_py()
    assert len(gray) == 4 * 4  # not *3
    assert all(v == 76 for v in gray)
    assert out.column("g")[1].as_py() is None
    # Shaped as a single-channel fixed-shape tensor.
    ttype = str(out.schema.field("g").type)
    assert ttype.startswith("extension<arrow.fixed_shape_tensor")
    assert "shape=[4,4,1]" in ttype


def test_image_decode_over_multicolumn_source(tmp_path):
    """Decoding a column of a multi-column source: projection pushdown must keep the
    decoded input column (regression — `referenced_columns` must traverse ImageFunc).
    """
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i, (w, h) in enumerate([(4, 4), (8, 8), (2, 2)]):
        (img_dir / f"i{i}.png").write_bytes(_png(w, h))

    out = bt.read.images(str(img_dir)).select(d=bt.col("bytes").image.decode()).collect()
    dims = sorted((r["width"], r["height"]) for r in out.column("d").to_pylist())
    assert dims == [(2, 2), (4, 4), (8, 8)]


def test_image_func_ir_shape():
    expr = bt.col("img").image.to_tensor(224, 224)
    ir = expr.to_ir()
    assert ir["e"] == "image"
    assert ir["fn"] == "to_tensor"
    assert ir["width"] == 224
    assert ir["height"] == 224


@pytest.mark.parametrize("fn", ["decode", "to_tensor"])
def test_image_null_bytes_yield_null(fn):
    ds = bt.from_arrow(pa.table({"img": pa.array([None], type=pa.binary())}))
    expr = bt.col("img").image.decode() if fn == "decode" else bt.col("img").image.to_tensor(2, 2)
    out = ds.select(r=expr).collect()
    assert out.column("r")[0].as_py() is None
