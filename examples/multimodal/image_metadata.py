"""Filtering images on metadata before paying to decode them.

Width, height, size and mime come from the file header, so a filter on them costs no
decode. Putting that filter first is the single biggest win in a multimodal pipeline,
because decoding is orders of magnitude more expensive than reading a header.

    python examples/multimodal/image_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import images
from batcher import col


def main() -> None:
    pictures = bt.read.images(images(100))
    print("images:", pictures.count())
    assert pictures.count() == 100

    described = pictures.select("uri", "size", "mime", "width", "height").with_columns(
        pixels=col("width") * col("height"),
        aspect=col("width") / col("height"),
    )

    summary = described.agg(
        smallest=col("size").min(),
        largest=col("size").max(),
        mean_pixels=col("pixels").mean(),
    ).to_pydict()
    print({name: round(value[0], 1) for name, value in summary.items()})

    rows = described.head(3).to_pydict()
    print(rows["mime"], rows["width"], rows["height"])
    assert all(value == "image/jpeg" for value in described.to_pydict()["mime"])
    assert summary["smallest"][0] > 0

    # The cheap filters: shape and size, no decode.
    square_ish = described.filter((col("aspect") > 0.9) & (col("aspect") < 1.1))
    print("near-square images:", square_ish.count())
    assert square_ish.count() <= pictures.count()

    big_enough = described.filter(col("pixels") > 1_000)
    assert big_enough.count() <= pictures.count()

    # Only now is decoding worth it, and only for what survived. A decode needs a
    # target size, because the batch has to come back as one rectangular tensor.
    assert bt.read.images(images(10), decode=True, size=(64, 64)).count() == 10


if __name__ == "__main__":
    main()
