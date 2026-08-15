"""Reading prose out of the document formats a corpus actually contains.

`DocumentSource` used to read PDFs and nothing else, which is a strange place for a
retrieval pipeline to stop: the documents an organization has are PDFs, web pages, Word
files, decks, and Markdown, in roughly that order of volume and exactly the reverse order
of how hard they are to read. A corpus that mixes them had to be split by extension outside
the engine and re-joined afterwards.

Every extractor here returns **a list of page texts**, because that is the shape the
`{path, page, text}` schema already has and because each format has a real notion of a
page: a PDF page, a slide, an EPUB spine item. Formats with no pagination — HTML, Markdown,
Word — return a single entry, so ``page`` is 0 and the column keeps meaning "which part of
the document this text came from" rather than becoming a per-format lie.

**Nothing here adds a dependency.** DOCX, PPTX and EPUB are ZIP archives of XML, and HTML
has a parser in the standard library, so all four are read with `zipfile`, `xml.etree` and
`html.parser`. That is deliberate and it is the same trade `bc-secrets` makes: this module
sits under a reader that a plain text pipeline also imports, and pulling a document-parsing
stack into every install to serve it would be paid by everyone. PDF keeps its `pypdf`
dependency because a PDF is genuinely not a text format.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable
from html.parser import HTMLParser

from batcher._internal.errors import IOError as BatcherIOError

__all__ = ["PAGE_EXTRACTORS", "extractor_for", "html_to_text"]

# Elements whose *content* is code for the browser rather than prose for the reader. Their
# text is dropped entirely: a page whose extracted text is mostly a minified analytics
# bundle is worse than no text, because it looks like a successful read.
_NON_PROSE = frozenset({"script", "style", "noscript", "template", "svg"})

# Elements that end a line of prose. Without them every heading, paragraph and list item on
# a page runs into the next word, so ``<h1>Title</h1><p>Body`` extracts as ``TitleBody`` --
# which changes the sentence and paragraph counts every downstream text metric reads.
_BLOCK = frozenset(
    {
        "p", "div", "br", "li", "tr", "td", "th", "section", "article", "header", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "table", "hr",
    }
)  # fmt: skip

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


class _TextExtractor(HTMLParser):
    """Collect an HTML document's readable text, dropping scripts, styles and markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:  # noqa: ARG002
        """Enter a tag: start suppressing non-prose content, or break the line."""
        if tag in _NON_PROSE:
            self._suppress += 1
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Leave a tag: stop suppressing, or break the line."""
        if tag in _NON_PROSE:
            # Clamped at zero because real-world HTML closes tags it never opened, and a
            # negative depth would suppress the entire rest of the document.
            self._suppress = max(0, self._suppress - 1)
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Keep character data unless it belongs to a non-prose element."""
        if self._suppress == 0:
            self._parts.append(data)

    def text(self) -> str:
        """The collected prose, with runs of blank lines collapsed."""
        joined = "".join(self._parts)
        joined = _TRAILING_SPACE.sub("\n", joined)
        return _BLANK_RUN.sub("\n\n", joined).strip()


def html_to_text(markup: str) -> str:
    """Extract readable text from an HTML document.

    Shared with the EPUB reader, whose chapters *are* XHTML documents — one extractor, so a
    web page and a book chapter cannot come out formatted differently.

    Args:
        markup: The HTML source.

    Returns:
        The document's text with tags removed, block elements separated by newlines, and
        script/style content dropped.
    """
    parser = _TextExtractor()
    # `HTMLParser` raises only on inputs it cannot tokenize at all; a malformed document is
    # ordinary on the web, so whatever was collected before the failure is the answer.
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # a broken page still has whatever text was read before it
        pass
    return parser.text()


def _decode(data: bytes) -> str:
    """Decode document bytes as text, tolerating a corpus that is not all UTF-8.

    A scraped corpus is not one encoding. UTF-8 first because it is nearly everything and
    because a successful strict decode is proof; then UTF-8 with a BOM, which Windows
    editors write and which leaves a stray ``\\ufeff`` on the first word otherwise; then
    cp1252, which decodes every byte sequence and so is the terminal fallback. Never
    ``errors="ignore"``: dropping the bytes it cannot read is how a page silently loses
    every accented character.
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # cp1252 maps every byte, so this is unreachable for `bytes`; the replacement decode is
    # here so a future encoding list that does not end in a total codec still cannot raise.
    return data.decode("utf-8", errors="replace")


def _plain(data: bytes) -> list[str]:
    """Markdown and plain text: the file *is* the document text, decoded and unmodified.

    Deliberately not rendered to prose. Markdown's markup is meaningful to a retrieval
    pipeline — a heading is a section boundary and a fenced block is code — and the string
    namespace already has `strip_html`, `remove_markdown_links` and the rest for a caller
    who wants it stripped. Rendering here would throw that away before anyone could choose.
    """
    return [_decode(data)]


def _html(data: bytes) -> list[str]:
    """One HTML file, extracted to a single page of prose."""
    return [html_to_text(_decode(data))]


def _zip_member_text(archive: zipfile.ZipFile, name: str, tags: frozenset[str]) -> str:
    """Concatenate the text of the named XML elements in one archive member.

    The shape DOCX and PPTX share: the payload is an XML part whose runs of text live in a
    single element type, namespaced. Matching on the *local* name means the extractor does
    not have to carry the OOXML namespace URIs, which differ between the two formats and
    between the strict and transitional variants of each.
    """
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(archive.read(name))
    except ElementTree.ParseError:
        return ""
    parts: list[str] = []
    for element in root.iter():
        local = element.tag.rpartition("}")[2]
        if local in tags and element.text:
            parts.append(element.text)
        # A paragraph break in OOXML is the *end* of a `<w:p>` / `<a:p>` element, not a
        # character in the text, so without this every paragraph of a Word document runs
        # into the next one exactly as unseparated HTML blocks do.
        elif local == "p":
            parts.append("\n")
    return _BLANK_RUN.sub("\n\n", "".join(parts)).strip()


def _open_archive(data: bytes, kind: str) -> zipfile.ZipFile:
    """Open a ZIP-backed document, or raise a `BatcherIOError` naming the format."""
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise BatcherIOError(f"not a readable {kind} file: {exc}") from exc


def _docx(data: bytes) -> list[str]:
    """A Word document, as one page.

    Word has no pagination in the file: page breaks are computed by the renderer from the
    fonts and the paper size, so a `page` column derived here would be invented. One entry
    is the honest answer, and a caller who wants smaller units has `str.chunk`.
    """
    with _open_archive(data, "docx") as archive:
        return [_zip_member_text(archive, "word/document.xml", frozenset({"t"}))]


def _pptx(data: bytes) -> list[str]:
    """A slide deck, one row per slide.

    The one format whose pages are genuinely free: `ppt/slides/slideN.xml` *is* the
    pagination, so the `page` column means the slide number a reader would cite.
    """
    with _open_archive(data, "pptx") as archive:
        slides = sorted(
            (n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            # Numeric, not lexicographic: `slide10.xml` sorts before `slide2.xml` as text,
            # which silently reorders every deck with ten or more slides.
            key=lambda n: int(re.search(r"(\d+)", n.rpartition("/")[2]).group(1)),  # type: ignore[union-attr]
        )
        return [_zip_member_text(archive, name, frozenset({"t"})) for name in slides]


def _epub(data: bytes) -> list[str]:
    """An EPUB book, one row per spine item, in reading order.

    Reading order comes from the OPF spine rather than from the archive's member order,
    which is arbitrary — a book read in ZIP order is a book with its chapters shuffled.
    A book whose manifest cannot be parsed falls back to sorted XHTML members, which is
    wrong less often than it is useless.
    """
    from xml.etree import ElementTree

    with _open_archive(data, "epub") as archive:
        names = _epub_spine(archive) or sorted(
            n for n in archive.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))
        )
        pages = []
        for name in names:
            try:
                pages.append(html_to_text(_decode(archive.read(name))))
            except (KeyError, ElementTree.ParseError):
                # A spine entry pointing at a missing member is a broken book, not a
                # broken corpus: the chapter is empty and the rest of the book is read.
                pages.append("")
        return pages


def _epub_spine(archive: zipfile.ZipFile) -> list[str]:
    """The archive members named by the OPF spine, in reading order (empty if unreadable)."""
    from xml.etree import ElementTree

    try:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(
            e for e in container.iter() if e.tag.rpartition("}")[2] == "rootfile"
        ).attrib["full-path"]
        opf = ElementTree.fromstring(archive.read(rootfile))
    except Exception:  # any malformed part means "no spine", not a failed read
        return []
    base = rootfile.rpartition("/")[0]
    # The manifest maps an id to a path; the spine lists ids in reading order.
    manifest = {
        e.attrib["id"]: e.attrib["href"]
        for e in opf.iter()
        if e.tag.rpartition("}")[2] == "item" and "id" in e.attrib and "href" in e.attrib
    }
    order = [
        e.attrib["idref"]
        for e in opf.iter()
        if e.tag.rpartition("}")[2] == "itemref" and "idref" in e.attrib
    ]
    members = set(archive.namelist())
    paths = []
    for ref in order:
        href = manifest.get(ref)
        if href is None:
            continue
        full = f"{base}/{href}" if base else href
        if full in members:
            paths.append(full)
    return paths


#: Extension → the extractor that turns one file's bytes into its list of page texts.
#: PDF is absent on purpose: it streams page by page through `pypdf` so a thousand-page
#: volume never has all of its text resident, which a bytes-in function cannot express.
PAGE_EXTRACTORS: dict[str, Callable[[bytes], list[str]]] = {
    ".htm": _html,
    ".html": _html,
    ".xhtml": _html,
    ".md": _plain,
    ".markdown": _plain,
    ".rst": _plain,
    ".txt": _plain,
    ".docx": _docx,
    ".pptx": _pptx,
    ".epub": _epub,
}


def extractor_for(path: str) -> Callable[[bytes], list[str]] | None:
    """The extractor for a path's extension, or `None` when it is the streaming PDF path.

    Args:
        path: The file path or URI being read.

    Returns:
        The bytes-to-pages extractor, or `None` for a `.pdf` (which `DocumentSource` reads
        through its own streaming reader).
    """
    return PAGE_EXTRACTORS.get("." + path.rpartition(".")[2].lower())
