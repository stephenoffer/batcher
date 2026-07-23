"""Generated benchmark datasets — the one place the suite builds data instead of reading it.

Every other dataset is read from canonical public parquet (``sources``); the semistructured
suite has no public nested-JSON corpus to point at, so it generates one deterministically and
shares the byte-identical Arrow table across every engine (the same parity the loaders give).
"""

from __future__ import annotations

from datagen.json_events import build_events

__all__ = ["build_events"]
