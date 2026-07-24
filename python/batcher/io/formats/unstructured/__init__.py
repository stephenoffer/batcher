"""`io.formats.unstructured` — sources with no schema beyond bytes or text.

Plain text, binary blobs, and documents have no records and no fields to infer, so
these sources produce a deliberately minimal Arrow schema — little more than a path
alongside the content — and leave all structure to be extracted downstream by
expressions: `.str`, `.json`, and the multimodal decoders. That is the line between
this family and `semistructured`: there, a record structure exists and is inferred;
here, there is none to find, so inventing one at read time would be guessing.

Because there is nothing to split on *inside* a shapeless file, these split at
whole-file granularity rather than by byte range — worth knowing before you reach
for parallelism here, since a single huge file is a single unit of work.

A new format whose content is opaque at read time is one new module here.
"""

from __future__ import annotations

from batcher.io.formats.unstructured.binary import BinarySource
from batcher.io.formats.unstructured.documents import DocumentSource
from batcher.io.formats.unstructured.text import TextSource
from batcher.io.formats.unstructured.warc import WARC_SCHEMA, WarcSource

__all__ = ["WARC_SCHEMA", "BinarySource", "DocumentSource", "TextSource", "WarcSource"]
