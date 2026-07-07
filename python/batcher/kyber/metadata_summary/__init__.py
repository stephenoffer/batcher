"""EXACT-gated per-column *summary* metadata shortcuts (façade).

Re-exports the public answer functions; the implementation (and its provenance firewall)
lives in `answers`. Packaged as a directory only to keep `kyber/` within its file-count
budget — the public import path `batcher.kyber.metadata_summary` is unchanged.
"""

from __future__ import annotations

from batcher.kyber.metadata_summary.answers import (
    answer_column_summary,
    approx_column_summary,
)

__all__ = [
    "answer_column_summary",
    "approx_column_summary",
]
