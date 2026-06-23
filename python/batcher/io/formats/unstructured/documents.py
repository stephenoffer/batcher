"""Document format — PDF text extraction via `pypdf`, to Arrow.

`DocumentSource` extracts text from PDF files into the Arrow schema
``{path: str, page: int64, text: str}`` — one row per page, assembled at batch
granularity (the unavoidable extraction for a non-tabular source). This is the
ingest path for RAG / document-AI pipelines; downstream chunking, embedding, and
search run as Rust expressions over the ``text`` column. Read-only; one file is one
`Split`.

All `pypdf` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[pdf]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["DocumentSource"]

#: The fixed schema every `DocumentSource` produces (one row per page).
DOCUMENT_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("page", pa.int64()),
        ("text", pa.string()),
    ]
)


def _require_pypdf() -> Any:
    """Import and return the `pypdf` module or raise `BackendError`."""
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("PDF support requires pypdf: pip install 'batcher[pdf]'") from exc
    return pypdf


@SOURCES.register("documents")
class DocumentSource(FileSource):
    """One or more PDF documents, one Arrow row per page.

    Produces ``{path: str, page: int64, text: str}`` — the per-page extracted text,
    ready for downstream Rust chunking/embedding over the ``text`` column.
    """

    suffix = ".pdf"
    format_name = "documents"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return DOCUMENT_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        pypdf = _require_pypdf()
        name = getattr(fh, "name", self._path)
        try:
            reader = pypdf.PdfReader(fh)
        except Exception as exc:
            raise BackendError(f"failed to read PDF {name!r}: {exc}") from exc
        paths: list[str] = []
        pages: list[int] = []
        texts: list[str] = []
        for number, page in enumerate(reader.pages):
            paths.append(name)
            pages.append(number)
            texts.append(page.extract_text() or "")
        batch = pa.RecordBatch.from_pydict(
            {"path": paths, "page": pages, "text": texts}, schema=DOCUMENT_SCHEMA
        )
        return [batch.select(projection) if projection is not None else batch]
