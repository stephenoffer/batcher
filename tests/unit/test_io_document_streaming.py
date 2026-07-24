"""`DocumentSource` streams pages, and each row names its own document.

Two problems, one fix. The reader built a single batch holding *every* page's extracted
text, so a scanned volume was resident whole — and page counts are unbounded, which is
exactly the shape a document corpus has.

And it labelled every row with ``getattr(fh, "name", self._path)``. The filesystem handle
carries no ``name``, so the fallback fired every time and a read of a directory stamped
every page of every document with the **directory** path. The `path` column exists to say
which document a page came from; it said the same thing for all of them. Nothing about
that is visible in a row count.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pypdf = pytest.importorskip("pypdf")

import batcher.io.formats.unstructured.documents as doc_mod  # noqa: E402
from batcher._internal.errors import FormatError  # noqa: E402
from batcher.io.formats.unstructured.documents import DocumentSource  # noqa: E402

pytestmark = pytest.mark.unit


def _pdf(path, pages: int) -> None:
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as fh:
        writer.write(fh)


@pytest.fixture
def corpus(tmp_path):
    """Two documents, so a per-document label is distinguishable from a per-directory one."""
    _pdf(tmp_path / "a.pdf", 600)
    _pdf(tmp_path / "b.pdf", 600)
    return str(tmp_path)


def test_every_row_names_its_own_document(corpus) -> None:
    table = pa.Table.from_batches(list(DocumentSource(corpus).iter_batches()))
    names = {p.rsplit("/", 1)[-1] for p in table.column("path").to_pylist()}

    assert names == {"a.pdf", "b.pdf"}, "pages are not attributed to their document"


def test_read_and_iter_batches_agree(corpus) -> None:
    """They are now one reader; before, they disagreed on the `path` column."""
    src = DocumentSource(corpus)

    assert pa.Table.from_batches(src.read()).equals(pa.Table.from_batches(list(src.iter_batches())))


def test_pages_are_emitted_in_several_batches(corpus) -> None:
    """One batch for a whole document is the thing being fixed, so assert the batching."""
    batches = list(DocumentSource(corpus).iter_batches())

    assert len(batches) > 2
    assert sum(b.num_rows for b in batches) == 1200


def test_page_numbers_restart_per_document(corpus) -> None:
    table = pa.Table.from_batches(list(DocumentSource(corpus).iter_batches()))
    rows = zip(table.column("path").to_pylist(), table.column("page").to_pylist(), strict=True)

    per_doc: dict[str, list[int]] = {}
    for path, page in rows:
        per_doc.setdefault(path, []).append(page)
    for path, pages in per_doc.items():
        assert pages == list(range(600)), f"{path} page numbers are not 0..599"


def test_a_small_document_still_yields_one_batch(tmp_path) -> None:
    _pdf(tmp_path / "tiny.pdf", 3)
    batches = list(DocumentSource(str(tmp_path)).iter_batches())

    assert sum(b.num_rows for b in batches) == 3


def test_projection_is_honored(corpus) -> None:
    batch = next(iter(DocumentSource(corpus).iter_batches(["page"])))
    assert batch.schema.names == ["page"]


def test_a_page_that_will_not_extract_is_null_under_skip(corpus, monkeypatch) -> None:
    """One bad page in a 900-page report must not cost the other 899."""

    def explode(self):
        raise ValueError("cannot extract")

    monkeypatch.setattr(pypdf._page.PageObject, "extract_text", explode)
    src = DocumentSource(corpus, on_error="skip")
    table = pa.Table.from_batches(list(src.iter_batches()))

    assert table.num_rows == 1200, "rows were dropped rather than nulled"
    assert table.column("text").null_count == 1200
    assert src.corrupt_files(), "the failure left no audit trail"


def test_a_page_that_will_not_extract_raises_by_default(corpus, monkeypatch) -> None:
    """Without `on_error`, an extraction failure propagates — as a *typed* error.

    `ErrorPolicy` raises `FormatError` rather than re-raising the backend's own exception,
    so the message names the file and the flag that would tolerate it instead of leaving
    the caller with whichever error the backend happened to pick. The original is chained,
    so nothing is lost — asserted here, because a wrapper that swallowed the cause would
    be worse than the bare error it replaced.
    """

    def explode(self):
        raise ValueError("cannot extract")

    monkeypatch.setattr(pypdf._page.PageObject, "extract_text", explode)

    with pytest.raises(FormatError) as exc:
        list(DocumentSource(corpus).iter_batches())
    assert "on_error='skip'" in str(exc.value)
    assert isinstance(exc.value.__cause__, ValueError)
    assert "cannot extract" in str(exc.value.__cause__)


def test_the_batch_size_is_what_splits_the_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doc_mod, "_PAGES_PER_BATCH", 10)
    _pdf(tmp_path / "a.pdf", 25)

    batches = list(DocumentSource(str(tmp_path)).iter_batches())

    assert [b.num_rows for b in batches] == [10, 10, 5]
