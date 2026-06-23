"""Video source — list video files + header-only fps/frames/width/height/duration.

`VideoSource` lists video files and emits the common media columns
(``uri, bytes, size, mime``) plus ``fps, frames, width, height, duration`` read
from each file's *container header* via PyAV. Opening a container and reading a
stream's metadata parses the header/index only — no video frame is decoded.
``duration`` is seconds; ``frames`` is the stream's declared frame count (may be
0/None when the container does not record it).

Needs the ``video`` extra (PyAV): ``pip install 'batcher[video]'``.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["VideoSource"]


@SOURCES.register("video")
class VideoSource(MediaSource):
    """One or more video files (directory or glob) as references + header meta."""

    suffixes = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
    format_name = "video"

    __slots__ = ()

    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("fps", pa.float64()),
            ("frames", pa.int64()),
            ("width", pa.int64()),
            ("height", pa.int64()),
            ("duration", pa.float64()),
        ]

    def _extract_meta(self, data: bytes) -> dict[str, Any]:
        av = _av()
        # Opening the container parses its header; reading stream attributes does
        # not decode any frame.
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            duration = float(container.duration / av.time_base) if container.duration else None
            return {
                "fps": fps,
                "frames": int(stream.frames) if stream.frames else None,
                "width": int(stream.width) if stream.width else None,
                "height": int(stream.height) if stream.height else None,
                "duration": duration,
            }


def _av() -> Any:
    """The PyAV module, or a typed error pointing at the ``video`` extra."""
    try:
        import av
    except ImportError as exc:
        raise BackendError(
            "reading video needs the video extra: pip install 'batcher[video]'"
        ) from exc
    return av
