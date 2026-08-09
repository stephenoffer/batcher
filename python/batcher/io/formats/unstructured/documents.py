"""Document format — PDF text extraction via `pypdf`, to Arrow.

`DocumentSource` extracts text from PDF files into the Arrow schema
``{path: str, page: int64, text: str}`` — one row per page, assembled at batch
granularity (the unavoidable extraction for a non-tabular source). This is the
ingest path for RAG / document-AI pipelines; downstream chunking, embedding, and
search run as Rust expressions over the ``text`` column. Read-only; one file is one
`Split`.

All `pypdf` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[pdf]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.optional import require
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["DocumentSource"]

#: The fixed schema every `DocumentSource` produces (one row per page).
# Pages per emitted batch. A document is not bounded — a scanned volume can run to tens of
# thousands of pages — so the extracted text is handed downstream in batches rather than
# accumulated whole.
_PAGES_PER_BATCH = 512

DOCUMENT_SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("page", pa.int64()),
        ("text", pa.string()),
    ]
)


def _require_pypdf() -> Any:
    """Import and return the `pypdf` module or raise `BackendError`."""
    return require("pypdf", feature="PDF support", provides="pypdf", extra="pdf")


@SOURCES.register("documents")
class DocumentSource(FileSource):
    """One or more PDF documents, one Arrow row per page.

    Produces ``{path: str, page: int64, text: str}`` — the per-page extracted text,
    ready for downstream Rust chunking/embedding over the ``text`` column.
    """

    suffix = ".pdf"
    format_name = "documents"

    __slots__ = ("_password",)

    def __init__(self, path: str, *, password: str | None = None, **kwargs: Any) -> None:
        """Open a PDF corpus, optionally supplying the password its files are locked with.

        Args:
            path: A PDF file, directory, or glob.
            password: The user password for an encrypted corpus. A PDF encrypted for
                *permissions* only carries an empty user password and opens without this;
                one with a real password could not be read at all before, since there was
                nowhere to put it.
            kwargs: Forwarded to `FileSource` (``on_error``, ``schema_mode``, ...).
        """
        super().__init__(path, **kwargs)
        self._password = password

    def _reader_kwargs(self) -> dict[str, object]:
        # A worker rebuilds the reader from these, and without the password every
        # encrypted file in the corpus would fail there while succeeding single-node —
        # a distributed result that differs from the local one.
        return {**super()._reader_kwargs(), "password": self._password}

    def _open_reader(self, pypdf: Any, source: Any, name: str) -> Any:
        """A `PdfReader` over `source`, decrypted when the file is locked.

        pypdf opens a permissions-only PDF on its own (its user password is empty), so
        this fires exactly for the files that genuinely need a secret. Failing here with
        the parameter's name is the point: the underlying error is
        ``FileNotDecryptedError: File has not been decrypted``, which tells a caller what
        happened and not one thing about what to do next.
        """
        try:
            reader = pypdf.PdfReader(source)
        except Exception as exc:
            raise BackendError(f"failed to read PDF {name!r}: {exc}") from exc
        if not reader.is_encrypted:
            return reader
        try:
            opened = reader.decrypt(self._password or "")
        except Exception as exc:
            raise BackendError(f"failed to decrypt PDF {name!r}: {exc}") from exc
        if not opened:
            raise BackendError(
                f"PDF {name!r} is encrypted and the password did not open it; pass the "
                "user password as `bt.read.documents(path, password=...)`"
            )
        return reader

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return DOCUMENT_SCHEMA

    def _read_by_path(self, path: str, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """Read a document by path, so `read()` and `iter_batches()` share one reader.

        Taking the path rather than a handle also fixes what the handle could not know:
        `_read_file` labelled every row with ``getattr(fh, "name", self._path)``, and the
        filesystem handle carries no ``name`` — so a read of a *directory* stamped every
        page of every document with the directory path. The `path` column exists to say
        which document a page came from, and it said the same thing for all of them.
        """
        return list(self._iter_file(path, projection))

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read from an open handle — the `FileSource` fallback when no path is available.

        `_read_by_path` takes precedence for every read this source performs; this exists
        because the template requires it, and because a handle-only caller must still get
        pages rather than an error. Its `path` column is best-effort for the reason above.
        """
        pypdf = _require_pypdf()
        name = getattr(fh, "name", self._path)
        reader = self._open_reader(pypdf, fh, name)
        paths, pages, texts = [], [], []
        for number, page in enumerate(reader.pages):
            paths.append(name)
            pages.append(number)
            texts.append(page.extract_text() or "")
        return [self._page_batch(paths, pages, texts, projection)]

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream a document's pages in batches instead of accumulating all of its text.

        `_read_file` builds one batch holding every page, so a scanned volume's entire
        extracted text is resident at once — and a document corpus is exactly where that
        bites, since page counts are unbounded and vary by orders of magnitude.

        **Extraction is skipped entirely when `text` is not projected.** Laying out a page
        into reading order is essentially the whole cost of this reader — measured at ~90%
        of a 300-page document's read time — and it was being paid unconditionally, so
        `select("path", "page")` cost the same as reading the prose and a bare `count()`
        cost it too. The other two columns come from enumerating `reader.pages`, which
        only walks the page tree.

        A page that will not extract is a *page*-level failure, not a document-level one:
        under ``on_error="skip"`` its text is null (distinguishable from a genuinely empty
        page) and the document is recorded in `corrupt_files()`, rather than one bad page
        in a 900-page report costing the other 899.
        """
        pypdf = _require_pypdf()
        wants_text = projection is None or "text" in projection
        with self._fs.open(path) as fh:
            reader = self._open_reader(pypdf, fh, path)
            paths: list[str] = []
            pages: list[int] = []
            texts: list[str | None] = []
            for number, page in enumerate(reader.pages):
                text: str | None = None
                if wants_text:
                    try:
                        text = page.extract_text() or ""
                    except Exception as exc:
                        self._errors.tolerate(path, exc, format_name=self.format_name)
                        text = None
                paths.append(path)
                pages.append(number)
                texts.append(text)
                if len(paths) >= _PAGES_PER_BATCH:
                    yield self._page_batch(paths, pages, texts, projection)
                    paths, pages, texts = [], [], []
            if paths:
                yield self._page_batch(paths, pages, texts, projection)

    @staticmethod
    def _page_batch(
        paths: list[str],
        pages: list[int],
        texts: list[str | None],
        projection: list[str] | None,
    ) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_pydict(
            {"path": paths, "page": pages, "text": texts}, schema=DOCUMENT_SCHEMA
        )
        return batch.select(projection) if projection is not None else batch
