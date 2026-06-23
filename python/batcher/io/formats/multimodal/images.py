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

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["ImageSource"]


@SOURCES.register("images")
class ImageSource(MediaSource):
    """One or more image files (directory or glob) as references + header meta."""

    suffixes = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif")
    format_name = "images"

    __slots__ = ()

    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("width", pa.int64()),
            ("height", pa.int64()),
            ("mode", pa.string()),
        ]

    def _extract_meta(self, data: bytes) -> dict[str, Any]:
        image = _pil_image()
        with image.open(io.BytesIO(data)) as img:
            # `.size` / `.mode` are populated from the header; no `.load()` call,
            # so pixel data is never decoded.
            width, height = img.size
            return {"width": width, "height": height, "mode": img.mode}


def _pil_image() -> Any:
    """The PIL.Image module, or a typed error pointing at the ``image`` extra."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise BackendError(
            "reading images needs the image extra: pip install 'batcher-engine[image]'"
        ) from exc
    return Image
