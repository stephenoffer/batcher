"""What a spilled partition *is*: which tier holds it, and how big it is two ways.

Split out from the store so the writer and the store can both name these without an
import cycle, and so the one thing a caller keeps after a spill — the handle — is
readable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["SpillHandle", "SpillTier"]


class SpillTier(Enum):
    """Which storage tier a spilled partition lives on."""

    LOCAL = "local"  # Arrow IPC on local disk (NVMe)
    REMOTE = "remote"  # object storage via fsspec


@dataclass(frozen=True, slots=True)
class SpillHandle:
    """An opaque reference to one spilled partition (tier + path + sizes).

    `nbytes` is the **compressed, on-disk** size (what the local-budget accounting charges).
    `logical_nbytes` is the **uncompressed, in-memory** size of everything written — what a
    reducer must budget against before reading the bucket back into RAM, since a compressible
    bucket's on-disk size can be many times smaller than its resident footprint.
    `num_rows` is how many rows the bucket holds, so a reducer can size a merge (and a caller
    can detect a truncated bucket) without opening the file.
    """

    tier: SpillTier
    path: str
    nbytes: int
    logical_nbytes: int = 0
    num_rows: int = 0
