"""Multimodal sources — images/audio/video/embeddings as queryable Arrow columns.

Each source lists media files and produces ``uri/bytes/size/mime`` plus cheap,
header-only metadata; decoding to pixels/tensors is a downstream Rust expression,
never done at read time.
"""

from __future__ import annotations

from batcher.io.formats.multimodal.audio import AudioSource
from batcher.io.formats.multimodal.embeddings import EmbeddingSource
from batcher.io.formats.multimodal.images import ImageSource
from batcher.io.formats.multimodal.media import MediaSource
from batcher.io.formats.multimodal.video import VideoSource

__all__ = ["AudioSource", "EmbeddingSource", "ImageSource", "MediaSource", "VideoSource"]
