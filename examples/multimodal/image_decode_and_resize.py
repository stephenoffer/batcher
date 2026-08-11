"""Decoding image bytes, and resizing on the way in.

`decode=True` turns the bytes into pixels, and it *requires* a `size=`. That is not an
omission: a decoded batch has to be one rectangular tensor, so the target shape has to be
known before the first file is opened. Resizing during the decode also means no
full-resolution frame is ever materialized, which is what blows up memory otherwise.

    python examples/multimodal/image_decode_and_resize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import images
from batcher import col


def main() -> None:
    glob = images(10)

    # Header only.
    headers = bt.read.images(glob)
    print("header columns:", headers.columns)
    assert "bytes" in headers.columns

    # Decoding without a size is refused rather than guessed at.
    try:
        bt.read.images(glob, decode=True)
    except Exception as error:
        print("refused:", str(error)[:70])
    else:
        raise AssertionError("decode=True must require a size")

    decoded = bt.read.images(glob, decode=True, size=(64, 64))
    print("decoded columns:", decoded.columns)
    assert decoded.count() == headers.count()

    # Decoded and resized in one step, so no full-resolution frame is ever materialized.
    # The pixels land in a new `image` column; `width` and `height` keep describing the
    # *source* file, which is what you still want for filtering and provenance.
    resized = bt.read.images(glob, decode=True, size=(32, 32))
    assert "image" in resized.columns
    assert resized.count() == 10

    source_shape = resized.select("width", "height").to_pydict()
    print("source was:", source_shape["width"][0], "x", source_shape["height"][0])
    assert all(value != 32 for value in source_shape["width"])

    # The tensor itself is flat: 32 * 32 * 3 channels.
    tensor = resized.select("image").head(1).to_pydict()["image"][0]
    print("decoded tensor length:", len(tensor))
    assert len(tensor) == 32 * 32 * 3

    # A larger target gives a proportionally larger tensor, which is the check that the
    # `size` argument is doing the resizing rather than being ignored.
    bigger = bt.read.images(glob, decode=True, size=(64, 64))
    assert len(bigger.select("image").head(1).to_pydict()["image"][0]) == 64 * 64 * 3

    # Metadata survives the decode, so a filter still works downstream.
    assert resized.filter(col("width") > 0).count() == 10


if __name__ == "__main__":
    main()
