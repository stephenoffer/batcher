"""Image source — list image files + header-only width/height/mode.

`ImageSource` lists image files and emits the common media columns
(``uri, bytes, size, mime``) plus ``width, height, mode`` read from each image's
*header* via Pillow. ``PIL.Image.open`` parses only the header to expose `.size`
and `.mode`; pixel data is decoded lazily on access, which this source never
triggers — so no image is ever decoded at read time.

Needs the ``image`` extra (Pillow): ``pip install 'batcher-engine[image]'``.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["ImageSource"]


@SOURCES.register("images")
class ImageSource(MediaSource):
    """One or more image files (directory or glob) as references + header meta."""

    # The listing is by extension, so an extension this tuple does not name is a file the
    # source cannot see at all -- and the error a user gets is "no images files under
    # <path>", which reads as an empty directory rather than as an unlisted format. That is
    # what a corpus of iPhone photographs got: `.heic` is what every phone since 2017
    # writes. The modern container formats are listed even where Pillow needs a plugin to
    # decode them, because the rows are still worth having: `bytes`, `size` and `mime` come
    # from the read, and an unparseable header nulls that file's metadata rather than
    # dropping its row.
    suffixes = (
        ".png", ".jpg", ".jpeg", ".jfif", ".gif", ".bmp", ".webp", ".tiff", ".tif",
        ".heic", ".heif", ".avif", ".ico", ".jp2", ".j2k", ".ppm", ".pgm",
    )  # fmt: skip
    format_name = "images"

    __slots__ = ()

    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("width", pa.int64()),
            ("height", pa.int64()),
            ("mode", pa.string()),
            ("format", pa.string()),
        ]

    def _extract_meta(self, data: bytes) -> dict[str, Any]:
        image = _pil_image()
        with image.open(io.BytesIO(data)) as img:
            # `.size` / `.mode` / `.format` are populated from the header; no `.load()`
            # call, so pixel data is never decoded. `format` is the *container* the bytes
            # actually are, which is worth having beside `mime` because a corpus assembled
            # by content type is full of files whose extension and container disagree --
            # those rows decode fine and break whatever downstream step branched on the
            # name. It costs nothing: the parse that produced the size produced it too.
            width, height = img.size
            return {
                "width": width,
                "height": height,
                "mode": img.mode,
                "format": img.format.lower() if img.format else None,
            }


def _pil_image() -> Any:
    """The PIL.Image module, or a typed error pointing at the ``image`` extra."""
    return require("PIL", "Image", feature="Image support", provides="Pillow", extra="image")
