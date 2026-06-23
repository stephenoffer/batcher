"""Write results — the manifest a sink returns and a commit consumes.

A `WrittenFile` records one physically-written data file; a `WriteManifest`
collects them for a whole write. The manifest is what makes distributed writes
mergeable: each worker returns its `WrittenFile`s and the driver concatenates
them (a commutative merge) into one manifest, then performs a single commit
(publishing a directory marker for file sinks, or an atomic transaction-log
commit for lakehouse sinks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["WriteManifest", "WrittenFile"]


@dataclass(frozen=True, slots=True)
class WrittenFile:
    """One data file written by a sink."""

    path: str
    rows: int
    bytes: int
    partition_values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteManifest:
    """The set of files produced by a write, plus rolled-up totals."""

    files: tuple[WrittenFile, ...] = ()

    @property
    def total_rows(self) -> int:
        return sum(f.rows for f in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def num_files(self) -> int:
        return len(self.files)

    def merge(self, other: WriteManifest) -> WriteManifest:
        """Combine two manifests (used to roll up distributed writer results)."""
        return WriteManifest(files=self.files + other.files)
