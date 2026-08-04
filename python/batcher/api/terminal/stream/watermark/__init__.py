"""`stream.watermark` — the streaming operators bounded by *event time*.

`dedup` emits a key once and forgets it when the watermark passes it; `join` is the
stream-stream interval join whose buffers the watermark releases. `_state` holds the
three things they share: state compaction, the microsecond normalization every bound is
expressed in, and the retained-state cap. The import path
`batcher.api.terminal.stream.watermark` is unchanged across the split.
"""

from __future__ import annotations

from batcher.api.terminal.stream.watermark._state import (
    _MAX_STATE_CHUNKS as _MAX_STATE_CHUNKS,
)
from batcher.api.terminal.stream.watermark._state import (
    _check_stream_state as _check_stream_state,
)
from batcher.api.terminal.stream.watermark._state import (
    _compact as _compact,
)
from batcher.api.terminal.stream.watermark.dedup import stream_watermark_dedup
from batcher.api.terminal.stream.watermark.join import stream_stream_join

__all__ = ["stream_stream_join", "stream_watermark_dedup"]
