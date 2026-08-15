"""Video source — list video files + header-only fps/frames/width/height/duration.

`VideoSource` lists video files and emits the common media columns
(``uri, bytes, size, mime``) plus ``fps, frames, width, height, duration`` read
from each file's *container header* via PyAV. Opening a container and reading a
stream's metadata parses the header/index only — no video frame is decoded.
``duration`` is seconds; ``frames`` is the stream's declared frame count (may be
0/None when the container does not record it).

Needs the ``video`` extra (PyAV): ``pip install 'batcher-engine[video]'``.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["VideoSource"]


@SOURCES.register("video")
class VideoSource(MediaSource):
    """One or more video files (directory or glob) as references + header meta."""

    # An extension this tuple does not name is invisible to the listing, and the error
    # reads as an empty directory rather than as an unlisted container. Broadcast and
    # surveillance corpora are `.ts` and `.wmv`; phone captures before H.265 are `.3gp`.
    suffixes = (
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
        ".wmv", ".flv", ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
    )  # fmt: skip
    format_name = "video"

    __slots__ = ()

    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("fps", pa.float64()),
            ("frames", pa.int64()),
            ("width", pa.int64()),
            ("height", pa.int64()),
            ("duration", pa.float64()),
            # The two facts a video corpus is actually triaged on, and the two the header
            # parse above was already holding and throwing away. `codec` decides whether a
            # hardware decoder can take a clip -- a corpus that mixes H.264 and VP9 splits
            # in half on it -- and `has_audio` decides whether a speech stage runs at all.
            # Learning either of them otherwise meant opening every container a second time.
            ("codec", pa.string()),
            ("has_audio", pa.bool_()),
        ]

    def _extract_meta(self, data: bytes) -> dict[str, Any]:
        av = _av()
        # Opening the container parses its header; reading stream attributes does
        # not decode any frame.
        with av.open(io.BytesIO(data)) as container:
            duration = float(container.duration / av.time_base) if container.duration else None
            has_audio = bool(container.streams.audio)
            # A container with no video stream is ordinary, not corrupt: an audio-only
            # `.mp4` and an `.mkv` holding only subtitles both occur in real corpora.
            # Indexing `streams.video[0]` raised `IndexError` on them, which the reader
            # tolerates by nulling *every* metadata column — so an audio-only file became
            # indistinguishable from a truncated one. Reporting the dimensions as absent
            # and `has_audio` as true says what the file is.
            video = container.streams.video[0] if container.streams.video else None
            if video is None:
                return {
                    "fps": None,
                    "frames": None,
                    "width": None,
                    "height": None,
                    "duration": duration,
                    "codec": None,
                    "has_audio": has_audio,
                }
            return {
                "fps": float(video.average_rate) if video.average_rate else None,
                "frames": int(video.frames) if video.frames else None,
                "width": int(video.width) if video.width else None,
                "height": int(video.height) if video.height else None,
                "duration": duration,
                # `codec_context.name` is the decoder's own short name (`h264`, `vp9`,
                # `hevc`), which is the vocabulary every placement decision is written in.
                "codec": getattr(video.codec_context, "name", None),
                "has_audio": has_audio,
            }


def _av() -> Any:
    """The PyAV module, or a typed error pointing at the ``video`` extra."""
    return require("av", feature="Video support", provides="PyAV", extra="video")
