"""Template-Method base classes for file-backed sources and sinks.

`FileSource` centralizes everything shared across file formats — path/glob/
filesystem resolution, schema caching, multi-file concatenation, projection
plumbing, streaming, and split generation — so a concrete format is a tiny
subclass overriding only its per-file read primitives. `FileSink` does the same
for writers: atomic writes, Hive partitioning, and the per-file manifest. This is
the shared-code spine that keeps each `io/formats/<fmt>.py` small (the v2 antidote
to v1's duplicated, mixin-heavy readers).

The `Source`/`Sink` protocols themselves live in `io.source`/`io.sink`; these
bases structurally satisfy them. `Split` lives in `io.splits`.

The query backends have their own spine, `io.formats.sql._source_base.SingleResultQuerySource`,
and it lives there rather than here because it is written against SQL-string rewriting
(`push_down`, `schema_probe`) — format specifics this neutral package deliberately does not
know. Look for it there when you are adding a database connector rather than a file format.
"""

from __future__ import annotations

from batcher.io.base.sink import FileSink
from batcher.io.base.sink import _safe_size as _safe_size  # re-export: used by csv/json sinks
from batcher.io.base.source import FileSource

__all__ = ["FileSink", "FileSource"]
