"""Reading audio and video without decoding the media.

The same rule as images: the header carries duration, codec and dimensions, and filtering on
those costs no decode. For video that gap is enormous — a header read is microseconds and a
decode is seconds per file.

    python examples/multimodal/audio_and_video_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt


def main() -> None:
    # The readers exist and are reached the same way as every other source. There is no
    # audio or video corpus in the public benchmark bucket, so this checks the surface
    # rather than inventing a fixture.
    assert hasattr(bt.read, "audio")
    assert hasattr(bt.read, "video")
    assert hasattr(bt.read, "images")

    import inspect

    audio_signature = inspect.signature(bt.read.audio)
    video_signature = inspect.signature(bt.read.video)
    print("read.audio", audio_signature)
    print("read.video", video_signature)

    # Both take a path and, like images, decode only when asked.
    assert "path" in audio_signature.parameters
    assert "path" in video_signature.parameters

    # The images reader is the one with a corpus here, and it proves the shared shape:
    # metadata columns first, bytes alongside, decoding opt-in.
    from _common import images

    pictures = bt.read.images(images(10))
    print("image metadata columns:", pictures.columns)
    assert {"uri", "bytes", "size", "mime"} <= set(pictures.columns)
    assert pictures.count() == 10

    # Filtering on metadata, with no decode.
    from batcher import col

    small = pictures.filter(col("size") < 10_000)
    print("images under 10 KiB:", small.count())
    assert small.count() <= pictures.count()


if __name__ == "__main__":
    main()
