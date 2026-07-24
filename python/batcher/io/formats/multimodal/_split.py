"""One file-batch of a media source, as a worker-side locator.

Split out of `media` because it is the *distributed* half of the media read surface and
shares no state with `MediaSource` — it carries file paths, rebuilds a source from the
`SOURCES` registry on the worker, and delegates. Keeping it here also keeps `media.py`
inside the module size limit without allowlisting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batcher.io.formats.multimodal.media import MediaSource

__all__ = ["MediaSplit"]


@dataclass(frozen=True, slots=True)
class MediaSplit:
    """One file-batch of a media source, reconstructed on a worker via `SOURCES`.

    Carries only ``(format_name, files, with_meta)`` — a tuple of file-path
    locators, never data — so it pickles cheaply to a remote worker that then
    reads just its files directly from storage. Mirrors the `Split` read surface
    so a worker treats a split exactly like a source.
    """

    format_name: str
    files: tuple[str, ...]
    with_meta: bool
    materialize_bytes: bool = True

    @property
    def rows(self) -> int:
        """This split's exact row count — one row per file, known with no I/O.

        The distributed planner reads `rows` off a split to size its task fan-out and to
        weight the partition balance. Without it a media source looked *uncountable*: the
        fan-out fell back to a blunt worker count and every split weighed the same, so a
        split of 200 MB videos was balanced against one of thumbnails as if equal.
        """
        return len(self.files)

    def _source(self) -> MediaSource:
        """Rebuild a source restricted to this split's files (no re-listing)."""
        from batcher.io.formats.base import SOURCES

        cls = SOURCES.get(self.format_name)
        # Reuse the source's batch assembly but pin its file list to this split's
        # files; batch_files is set so the whole split assembles as one batch.
        src: MediaSource = cls(
            self.files[0],
            batch_files=len(self.files),
            with_meta=self.with_meta,
            materialize_bytes=self.materialize_bytes,
        )
        src._files_cache = list(self.files)
        return src

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._source().read(projection)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._source().iter_batches(projection)

    def row_count(self) -> int | None:
        return len(self.files)

    def identity(self) -> str:
        return f"{self.format_name}:{self.files[0]}+{len(self.files)}"
