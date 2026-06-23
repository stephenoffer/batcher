"""`read.images(decode=True)` — native decode to a fixed-shape tensor column.

The decode/resize runs in the Rust `image` kernel; the result is re-typed (zero-copy)
to a canonical ``(H, W, 3)`` fixed-shape-tensor column so it converts to a shaped
training tensor. Built on the same stdlib PNG generator as `test_image_expr` (no image
library needed to produce the bytes).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.io.formats.ml.tensor import is_tensor_column
from batcher.ml.decode import image_tensor_dataset


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    idat = zlib.compress(row * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _ds():
    return bt.from_pydict(
        {
            "id": [1, 2, 3],
            "bytes": [_png(8, 6, (255, 0, 0)), _png(8, 6, (0, 255, 0)), _png(8, 6, (0, 0, 255))],
        }
    )


def test_decode_produces_shaped_tensor_column():
    out = image_tensor_dataset(_ds(), size=(4, 4)).collect()
    assert "image" in out.column_names
    field = out.schema.field("image")
    assert is_tensor_column(field.type)
    assert field.type.shape == [4, 4, 3]
    nd = out.column("image").combine_chunks().to_numpy_ndarray()
    assert nd.shape == (3, 4, 4, 3)
    assert nd.dtype == np.uint8


def test_decode_pixels_are_correct():
    out = image_tensor_dataset(_ds(), size=(4, 4)).collect()
    nd = out.column("image").combine_chunks().to_numpy_ndarray()
    assert nd[0, 0, 0].tolist() == [255, 0, 0]  # red
    assert nd[1, 0, 0].tolist() == [0, 255, 0]  # green
    assert nd[2, 0, 0].tolist() == [0, 0, 255]  # blue


def test_read_images_decode_option(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        (img_dir / f"i{i}.png").write_bytes(_png(8, 8, color))
    out = bt.read.images(str(img_dir), decode=True, size=(4, 4)).collect()
    assert is_tensor_column(out.column("image"))
    assert out.column("image").combine_chunks().to_numpy_ndarray().shape == (2, 4, 4, 3)


def test_decode_requires_size():
    with pytest.raises(PlanError, match="size="):
        image_tensor_dataset(_ds(), size=None)


def test_decoded_column_converts_to_torch_tensor():
    torch = pytest.importorskip("torch")
    from batcher.ml.loader import column_to_tensor

    out = image_tensor_dataset(_ds(), size=(4, 4)).collect()
    t = column_to_tensor(out.column("image"))
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (3, 4, 4, 3)
