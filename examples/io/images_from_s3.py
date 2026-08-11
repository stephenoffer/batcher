"""Reading real images from object storage as a table of bytes.

An image read produces one row per file, with the bytes plus the metadata that can be had
without decoding: size, mime type, width and height. Decoding is opt-in because it is the
expensive part, and most pipelines filter on the metadata first.

    python examples/io/images_from_s3.py
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
    print("reading", glob)

    pictures = bt.read.images(glob)
    print(pictures.schema)

    result = pictures.select("uri", "size", "mime", "width", "height").to_pydict()
    print(result["mime"][0], result["width"][0], "x", result["height"][0])

    assert pictures.count() == 10
    assert all(mime == "image/jpeg" for mime in result["mime"])
    assert all(size > 0 for size in result["size"])
    assert all(
        width > 0 and height > 0
        for width, height in zip(result["width"], result["height"], strict=True)
    )

    # The bytes are there, and they really are JPEG: every JPEG starts with FF D8 FF.
    raw = pictures.select("bytes").head(1).to_pydict()["bytes"][0]
    assert raw[:3] == b"\xff\xd8\xff"

    # Metadata filtering costs no decode.
    big = pictures.filter(col("size") > 4096)
    print("images over 4 KiB:", big.count())
    assert big.count() <= pictures.count()


if __name__ == "__main__":
    main()
