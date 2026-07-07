"""EXACT-gated *filtered-count* metadata shortcuts (façade).

Re-exports the public answer functions; the implementation (and its provenance
firewall) lives in `answers`. Packaged as a directory only to keep `kyber/` within its
file-count budget — the public import path `batcher.kyber.metadata_filter_count` is
unchanged.
"""

from __future__ import annotations

from batcher.kyber.metadata_filter_count.answers import (
    answer_filter_any,
    answer_filter_count,
    answer_filter_is_empty,
)

__all__ = [
    "answer_filter_any",
    "answer_filter_count",
    "answer_filter_is_empty",
]
