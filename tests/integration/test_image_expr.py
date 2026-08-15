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


def test_image_decode_reads_the_header_facts():
    ds = bt.from_arrow(pa.table({"img": pa.array([_PNG_1x1], type=pa.binary())}))
    out = ds.select(dims=bt.col("img").image.decode()).collect()
    dims = out.column("dims")[0].as_py()
    # One header read yields all four facts; `_png` writes an 8-bit RGB image.
    assert dims == {"width": 1, "height": 1, "channels": 3, "mode": "RGB"}


def test_image_decode_fields_project_individually():
    # The reason `decode` returns a struct rather than four functions: a caller takes
    # the field it wants, and the header is still read once.
    ds = bt.from_arrow(pa.table({"img": pa.array([_PNG_1x1], type=pa.binary())}))
    decoded = bt.col("img").image.decode()
    out = ds.select(
        w=decoded.struct.field("width"),
        mode=decoded.struct.field("mode"),
    ).to_pydict()
    assert out == {"w": [1], "mode": ["RGB"]}


def test_image_crop_takes_the_named_window():
    # A 4x4 image cropped to its top-left 2x2 comes back at 2x2, through the engine.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    region = bt.col("img").image.crop(0, 0, 2, 2)
    out = ds.select(d=region.image.decode()).to_pydict()
    assert out["d"][0]["width"] == 2
    assert out["d"][0]["height"] == 2


def test_image_crop_clips_instead_of_padding():
    # `center_crop` pads; `crop` clips. A window larger than what remains at the offset
    # yields the smaller real region rather than inventing black pixels.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    region = bt.col("img").image.crop(3, 3, 10, 10)
    out = ds.select(d=region.image.decode()).to_pydict()
    assert out["d"][0] == {"width": 1, "height": 1, "channels": 4, "mode": "RGBA"}


def test_image_encode_changes_the_container():
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    out = ds.select(
        png=bt.col("img").image.encode("png").image.decode().struct.field("mode"),
        jpeg=bt.col("img").image.encode("jpeg").image.decode().struct.field("mode"),
    ).to_pydict()
    # PNG keeps whatever the source had; JPEG has no alpha, so RGBA flattens to RGB.
    assert out["jpeg"] == ["RGB"]
    assert out["png"] == ["RGB"]


def test_image_convert_changes_the_color_mode():
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(4, 4)], type=pa.binary())}))
    got = {
        mode: ds.select(
            m=bt.col("img").image.convert(mode).image.decode().struct.field("mode")
        ).to_pydict()["m"][0]
        for mode in ("L", "LA", "RGB", "RGBA")
    }
    # `decode` names the mode `convert` produced, so the two share one vocabulary.
    assert got == {"L": "L", "LA": "LA", "RGB": "RGB", "RGBA": "RGBA"}


def test_image_convert_and_to_grayscale_agree_on_luma():
    # Rec. 601 in both, so a pipeline can use either without the greys shifting.
    ds = bt.from_arrow(pa.table({"img": pa.array([_png(2, 2)], type=pa.binary())}))
    via_convert = ds.select(t=bt.col("img").image.convert("L").image.to_tensor(2, 2)).to_pydict()[
        "t"
    ][0]
    via_kernel = ds.select(t=bt.col("img").image.to_grayscale(2, 2)).to_pydict()["t"][0]
    # `to_tensor` re-expands L to RGB, so compare the luma against every channel.
    assert via_convert[0] == via_kernel[0]


def test_image_convert_rejects_an_unknown_mode():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="mode must be one of"):
        bt.col("img").image.convert("CMYK")


def test_image_encode_rejects_an_unwritable_format():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="format must be one of"):
        bt.col("img").image.encode("webp")


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


def test_image_ops_survive_streaming_and_multiple_batches():
    """Batching must not change what a per-row image kernel returns, or its column type.

    `collect()` hands the engine one batch; `iter_batches()` hands it several. The kernels
    are per-row, so the values cannot legitimately differ -- which is exactly why this is
    worth asserting rather than assuming, because the two things that *can* differ are
    invisible in the values. A decoded tensor carries its shape as Arrow extension
    metadata, and metadata is what a boundary drops; and a batch that happens to contain
    only unusable rows is the shape that makes a kernel widen a `Null` column, which the
    single-batch path may never produce.

    The fixture puts a null and an undecodable row in the middle so at least one batch is
    all-bad at a small batch size.
    """
    rows = [_png(4, 3), None, b"not an image", _png(2, 5), _png(6, 6), None]
    ds = bt.from_arrow(pa.table({"b": pa.array(rows, type=pa.binary())}))
    cases = {
        "tensor": lambda c: c.image.to_tensor(4, 4),
        "meta": lambda c: c.image.decode(),
        "bright": lambda c: c.image.brightness(),
        "hash": lambda c: c.image.dhash(),
        "encoded": lambda c: c.image.encode("png"),
    }
    for name, build in cases.items():
        query = ds.select(x=build(bt.col("b")))
        whole = query.collect()
        batches = list(query.iter_batches(batch_size=2))

        streamed = [row for batch in batches for row in batch.column("x").to_pylist()]
        assert streamed == whole.column("x").to_pylist(), name
        assert len(batches) > 1, f"{name}: batch_size=2 over six rows must span batches"
        for batch in batches:
            assert batch.schema.field("x").type == whole.schema.field("x").type, name
