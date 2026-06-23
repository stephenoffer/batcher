"""Unstructured formats (text, binary blobs, documents)."""

from __future__ import annotations

from batcher.io.formats.unstructured.binary import BinarySource
from batcher.io.formats.unstructured.documents import DocumentSource
from batcher.io.formats.unstructured.text import TextSource

__all__ = ["BinarySource", "DocumentSource", "TextSource"]
