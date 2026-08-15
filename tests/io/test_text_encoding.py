"""A text corpus is not one encoding, and the two read modes must agree about that.

The defect these cover: `mode="line"` and `mode="file"` gave *different answers* for the
same bytes. Line mode's Arrow-backed splitter replaced what it could not decode and
returned the row; file mode's Python decode raised `UnicodeDecodeError` and lost the whole
read. One source, one file, two behaviours — and the silent one was the default path for
the mode people use most.

Scraped pages, exported logs and anything a Windows editor touched are exactly the corpus
where this bites, so the agreement is asserted mode-by-mode rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_MODES = ("line", "file")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """One clean UTF-8 file and one cp1252 file, the ordinary scraped mixture."""
    (tmp_path / "good.txt").write_bytes(b"hello\nworld\n")
    (tmp_path / "legacy.txt").write_bytes("caf\xe9 na\xefve\n".encode("cp1252"))
    return tmp_path


def _texts(ds) -> list[str]:
    return sorted(ds.collect().to_pydict()["text"])


@pytest.mark.parametrize("mode", _MODES)
def test_both_modes_replace_undecodable_bytes_by_default(corpus: Path, mode: str) -> None:
    """The row survives with U+FFFD in it, whichever mode read it."""
    got = _texts(bt.read.text(str(corpus), mode=mode))
    assert any("�" in text for text in got), got
    assert any("hello" in text for text in got), "the clean file must still be read"


@pytest.mark.parametrize("mode", _MODES)
def test_both_modes_fail_the_file_under_strict(corpus: Path, mode: str) -> None:
    """`errors="strict"` is the caller asking to be told, and both modes must tell them."""
    with pytest.raises(Exception, match=r"decode|codec"):
        bt.read.text(str(corpus), mode=mode, errors="strict").collect()


@pytest.mark.parametrize("mode", _MODES)
def test_a_skipped_file_does_not_cost_the_corpus(corpus: Path, mode: str) -> None:
    """One unreadable file in a scrape is ordinary; losing the other million rows is not."""
    ds = bt.read.text(str(corpus), mode=mode, errors="strict", on_error="skip")
    got = _texts(ds)
    assert (
        [t.strip() for t in got] == ["hello\nworld"]
        if mode == "file"
        else got
        == [
            "hello",
            "world",
        ]
    )


@pytest.mark.parametrize("mode", _MODES)
def test_naming_the_encoding_decodes_it_properly(corpus: Path, mode: str) -> None:
    """Replacement is a fallback, not an answer: the real fix is saying what the bytes are."""
    got = " ".join(_texts(bt.read.text(str(corpus), mode=mode, encoding="cp1252")))
    assert "café naïve" in got
    assert "�" not in got


def test_the_two_modes_agree_on_which_files_survive(corpus: Path) -> None:
    """The property the disagreement violated, stated directly.

    Line mode returns one row per line and file mode one per file, so the row *counts*
    differ by construction — what must match is which files contributed rows at all.
    """
    per_mode = {
        mode: {
            row["path"].rsplit("/", 1)[-1]
            for row in bt.read.text(str(corpus), mode=mode).collect().to_pylist()
        }
        for mode in _MODES
    }
    assert per_mode["line"] == per_mode["file"] == {"good.txt", "legacy.txt"}


def test_a_skipped_file_is_reported_rather_than_silently_missing(tmp_path: Path) -> None:
    """A skipped file leaves no row, so the only way to tell a partial read is to ask."""
    from batcher.io.formats.unstructured.text import TextSource

    (tmp_path / "good.txt").write_bytes(b"fine\n")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00broken\n")
    src = TextSource(str(tmp_path), mode="file", errors="strict", on_error="skip")
    rows = [row for batch in src.iter_batches() for row in batch.to_pylist()]
    assert [r["text"] for r in rows] == ["fine\n"]
    assert [p.rsplit("/", 1)[-1] for p in src.corrupt_files()] == ["bad.txt"]


def test_an_unknown_error_handler_is_refused_by_the_codec(tmp_path: Path) -> None:
    """A typo in `errors` must fail at construction, not a million rows into the scan.

    Python looks a handler up only when it first has to *use* one, so an unvalidated typo
    reads a clean corpus perfectly and then fails on the first undecodable byte, from
    inside a decoder, with no mention of the parameter that named it.
    """
    (tmp_path / "a.txt").write_bytes(b"hi\n")
    with pytest.raises(Exception, match=r"replce"):
        bt.read.text(str(tmp_path), mode="file", errors="replce").collect()
