"""Audio source — list audio files + header-only sample_rate/channels/duration.

`AudioSource` lists audio files and emits the common media columns
(``uri, bytes, size, mime``) plus ``sample_rate, channels, duration`` read from
each file's *header* via ``soundfile.info``, which parses container/format
metadata only — it never decodes the PCM sample array. ``duration`` is seconds
(``frames / sample_rate``).

Needs the ``audio`` extra (soundfile): ``pip install 'batcher-engine[audio]'``.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa

from batcher._internal.mathx import safe_div
from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["AudioSource"]


@SOURCES.register("audio")
class AudioSource(MediaSource):
    """One or more audio files (directory or glob) as references + header meta."""

    # Extensions this tuple does not name are invisible to the listing, and the resulting
    # error reads as an empty directory. Opus in particular is what every modern voice
    # recording and every WebRTC capture is stored as.
    suffixes = (
        ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif",
        ".m4a", ".m4b", ".aac", ".wma", ".au", ".caf",
    )  # fmt: skip
    format_name = "audio"

    __slots__ = ()

    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("sample_rate", pa.int64()),
            ("channels", pa.int64()),
            ("duration", pa.float64()),
        ]

    def _extract_meta(self, data: bytes) -> dict[str, Any]:
        sf = _soundfile()
        # `soundfile.info` reads only the header (frames/samplerate/channels);
        # the audio samples are never read.
        info = sf.info(io.BytesIO(data))
        duration = safe_div(info.frames, info.samplerate, None)
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration": duration,
        }


def _soundfile() -> Any:
    """The soundfile module, or a typed error pointing at the ``audio`` extra."""
    return require("soundfile", feature="Audio support", provides="soundfile", extra="audio")
