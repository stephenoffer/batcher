"""The unified read/write namespace — `bt.read` (readers) and `ds.write` (sinks).

Split into `reader` (the `Reader` / `bt.read` surface) and `writer` (the `Writer` /
`ds.write` surface); this façade preserves the `batcher.api.io_namespace` import path
and its `Reader` / `Writer` / `read` exports.
"""

from __future__ import annotations

from batcher.api.io_namespace.reader import Reader, read
from batcher.api.io_namespace.writer import Writer

__all__ = ["Reader", "Writer", "read"]
