"""The document reader over the formats a real corpus mixes, not just PDF.

`DocumentSource` read PDFs and nothing else, so a directory of PDFs, web pages, Word
files, decks and Markdown had to be split by extension outside the engine. These cover the
five formats added alongside it, and specifically the places where a naive extractor is
wrong in a way that still returns text: script content read as prose, paragraphs run
together, slides ordered lexicographically, and an EPUB read in archive order.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import batcher as bt
from batcher.io.formats.unstructured._extract import html_to_text

pytestmark = pytest.mark.integration


def _docx(paragraphs: list[list[str]]) -> bytes:
    """A minimal Word document: each inner list is one paragraph's runs of text."""
    body = "".join(
        "<w:p>" + "".join(f"<w:r><w:t>{run}</w:t></w:r>" for run in runs) + "</w:p>"
        for runs in paragraphs
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>",
        )
    return buf.getvalue()


def _pptx(slide_texts: list[str]) -> bytes:
    """A minimal deck: one `ppt/slides/slideN.xml` per entry, numbered from 1."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, text in enumerate(slide_texts, start=1):
            z.writestr(
                f"ppt/slides/slide{i}.xml",
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
                ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>",
            )
    return buf.getvalue()


def _epub(chapters: dict[str, str], spine: list[str]) -> bytes:
    """A minimal book: `chapters` maps id to body text, `spine` gives the reading order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>',
        )
        items = "".join(f'<item id="{k}" href="{k}.xhtml"/>' for k in chapters)
        refs = "".join(f'<itemref idref="{k}"/>' for k in spine)
        z.writestr(
            "OEBPS/book.opf",
            '<package xmlns="http://www.idpf.org/2007/opf">'
            f"<manifest>{items}</manifest><spine>{refs}</spine></package>",
        )
        for name, body in chapters.items():
            z.writestr(f"OEBPS/{name}.xhtml", f"<html><body><p>{body}</p></body></html>")
    return buf.getvalue()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A directory holding one document of each supported non-PDF format."""
    (tmp_path / "page.html").write_text(
        "<html><head><style>p{color:red}</style><script>track()</script></head>"
        "<body><h1>Title</h1><p>First.</p><p>Second.</p></body></html>"
    )
    (tmp_path / "notes.md").write_text("# Heading\n\nSome *markdown*.\n")
    (tmp_path / "report.docx").write_bytes(_docx([["Hello ", "world"], ["Second para"]]))
    (tmp_path / "deck.pptx").write_bytes(_pptx([f"Slide {i}" for i in range(1, 12)]))
    (tmp_path / "book.epub").write_bytes(
        _epub({"c1": "Chapter one.", "c2": "Chapter two."}, spine=["c2", "c1"])
    )
    return tmp_path


def _by_file(rows: dict[str, list]) -> dict[str, list[str]]:
    """Group the extracted text by file name, keeping each document's page order."""
    out: dict[str, list[str]] = {}
    order = sorted(range(len(rows["path"])), key=lambda i: (rows["path"][i], rows["page"][i]))
    for i in order:
        out.setdefault(rows["path"][i].rsplit("/", 1)[-1], []).append(rows["text"][i])
    return out


def test_one_scan_reads_every_format_in_a_mixed_corpus(corpus: Path) -> None:
    """The point of the change: five formats, one `read.documents`, one schema."""
    got = _by_file(bt.read.documents(str(corpus)).collect().to_pydict())
    assert set(got) == {"page.html", "notes.md", "report.docx", "deck.pptx", "book.epub"}
    assert got["page.html"] == ["Title\n\nFirst.\n\nSecond."]
    assert got["notes.md"] == ["# Heading\n\nSome *markdown*.\n"]
    assert got["report.docx"] == ["Hello world\nSecond para"]


def test_script_and_style_content_is_dropped_rather_than_read_as_prose() -> None:
    """A page whose text is a minified analytics bundle looks like a successful read."""
    text = html_to_text("<style>.a{x:1}</style><script>var tracked = 1;</script><p>Real words.</p>")
    assert text == "Real words."


def test_block_elements_break_the_line_so_words_do_not_run_together() -> None:
    """`<h1>Title</h1><p>Body` must not extract as `TitleBody`.

    Running them together changes the sentence and paragraph counts every downstream text
    metric reads, while still producing text that looks plausible.
    """
    assert html_to_text("<h1>Title</h1><p>Body</p>").splitlines() == ["Title", "", "Body"]


def test_markdown_keeps_its_markup(tmp_path: Path) -> None:
    """Markdown's markup is signal: a heading is a section boundary, a fence is code.

    Rendering it to prose in the reader would throw that away before a caller could
    choose; the string namespace already has the strippers for anyone who wants it gone.
    """
    source = "# Heading\n\n```python\nx = 1\n```\n\n[link](https://example.com)\n"
    (tmp_path / "doc.md").write_text(source)
    rows = bt.read.documents(str(tmp_path)).collect().to_pydict()
    assert rows["text"] == [source]


def test_slides_are_numbered_the_way_a_reader_would_cite_them(corpus: Path) -> None:
    """`slide10.xml` sorts before `slide2.xml` as text, which reorders any deck over nine."""
    rows = bt.read.documents(str(corpus)).filter(bt.col("path").str.contains("deck")).collect()
    got = _by_file(rows.to_pydict())["deck.pptx"]
    assert got == [f"Slide {i}" for i in range(1, 12)]


def test_an_epub_is_read_in_spine_order_not_archive_order(corpus: Path) -> None:
    """A book read in ZIP order is a book with its chapters shuffled."""
    rows = bt.read.documents(str(corpus)).filter(bt.col("path").str.contains("epub")).collect()
    assert _by_file(rows.to_pydict())["book.epub"] == ["Chapter two.", "Chapter one."]


def test_page_counts_and_projections_agree_across_read_paths(corpus: Path) -> None:
    """`count()`, a projection, and a streaming read must see the same rows.

    The three take different code paths — `read`, `select`, `iter_batches` — and a reader
    whose page numbering is built per path is exactly where they drift apart.
    """
    ds = bt.read.documents(str(corpus))
    expected = 1 + 1 + 1 + 11 + 2  # html + md + docx + 11 slides + 2 spine items
    assert ds.count() == expected
    assert ds.select("path", "page").collect().num_rows == expected
    assert sum(b.num_rows for b in ds.iter_batches()) == expected


def test_a_document_that_will_not_parse_is_tolerated_when_asked(tmp_path: Path) -> None:
    """One corrupt file in a corpus must not cost the rest of it."""
    (tmp_path / "good.html").write_text("<p>Fine.</p>")
    (tmp_path / "broken.docx").write_bytes(b"this is not a zip archive")
    with pytest.raises(Exception, match="docx"):
        bt.read.documents(str(tmp_path)).collect()
    rows = bt.read.documents(str(tmp_path), on_error="skip").collect().to_pydict()
    assert rows["text"] == ["Fine."]


def test_a_document_encoded_as_cp1252_keeps_its_accented_characters(tmp_path: Path) -> None:
    """A scraped corpus is not one encoding, and `errors="ignore"` silently drops the rest."""
    (tmp_path / "legacy.html").write_bytes("<p>caf\xe9 na\xefve</p>".encode("cp1252"))
    rows = bt.read.documents(str(tmp_path)).collect().to_pydict()
    assert rows["text"] == ["café naïve"]
