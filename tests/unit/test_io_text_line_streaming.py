"""`TextSource(mode="line")` streams a file instead of holding it whole.

Line mode is the one shape where a single *file* is unbounded — a multi-GB log — and the
old path held three copies of it at once: the decoded text, a Python list of every line,
and one batch containing all of them. Measured on a 128 MB log: 748 MB peak, 13x the win
from streaming.

Reading in blocks introduces two hazards a whole-file `splitlines()` never had, and both
are silent — they produce *plausible* lines, not an error:

* a multi-byte character split across a block boundary;
* a trailing `\\r` that turns out to be the first half of a `\\r\\n`, which would otherwise
  report one line where the file has none.

So the tests below drive tiny block sizes on purpose, to put a boundary at every offset.
"""

from __future__ import annotations

import pytest

import batcher.io.formats.unstructured.text as text_mod
from batcher.io.formats.unstructured.text import TextSource

pytestmark = pytest.mark.unit

# Inputs whose line structure is easy to get subtly wrong.
_CASES = {
    "plain": "a\nb\nc\n",
    "no_trailing_newline": "a\nb\nc",
    "crlf": "a\r\nb\r\nc\r\n",
    "cr_only": "a\rb\rc",
    "mixed_terminators": "a\r\nb\nc\rd",
    "empty": "",
    "only_newlines": "\n\n\n",
    "blank_lines": "a\n\nb\n\n",
    "unicode": "café\nnaïve\n中文\n",
    "long_lines": ("x" * 3000 + "\n") * 5,
}


@pytest.fixture
def write(tmp_path):
    def _write(text: str) -> str:
        (tmp_path / "a.txt").write_text(text, encoding="utf-8")
        return str(tmp_path)

    return _write


def _lines(path: str) -> tuple[list[str], list[int]]:
    lines, numbers = [], []
    for batch in TextSource(path, mode="line").iter_batches():
        lines += batch.column("text").to_pylist()
        numbers += batch.column("line_number").to_pylist()
    return lines, numbers


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_str_splitlines(write, name: str) -> None:
    """`splitlines()` is the oracle — it is what the whole-file path used."""
    text = _CASES[name]
    lines, numbers = _lines(write(text))

    assert lines == text.splitlines()
    assert numbers == list(range(1, len(lines) + 1))


@pytest.mark.parametrize("block", [1, 2, 3, 5, 7, 16])
@pytest.mark.parametrize("name", sorted(_CASES))
def test_block_boundaries_do_not_change_the_lines(write, monkeypatch, block, name) -> None:
    """A boundary at every offset — mid-character and mid-CRLF included."""
    monkeypatch.setattr(text_mod, "_TEXT_BLOCK_BYTES", block)
    text = _CASES[name]

    assert _lines(write(text))[0] == text.splitlines()


def test_line_numbers_stay_contiguous_across_batches(write, monkeypatch) -> None:
    """The counter carries across batches; restarting it per batch would be invisible
    in any single batch."""
    monkeypatch.setattr(text_mod, "_TEXT_LINES_PER_BATCH", 100)
    path = write("".join(f"line{i}\n" for i in range(1000)))

    batches = list(TextSource(path, mode="line").iter_batches())
    numbers = [n for b in batches for n in b.column("line_number").to_pylist()]

    assert len(batches) == 10, "expected the input to span several batches"
    assert numbers == list(range(1, 1001))


def test_an_empty_file_still_yields_its_schema(write) -> None:
    batches = list(TextSource(write(""), mode="line").iter_batches())

    assert sum(b.num_rows for b in batches) == 0
    assert batches[0].schema.names == ["path", "line_number", "text"]


def test_projection_is_honored(write) -> None:
    path = write("a\nb\n")
    batch = next(iter(TextSource(path, mode="line").iter_batches(["text"])))

    assert batch.schema.names == ["text"]


def test_file_mode_is_unchanged(write) -> None:
    """Only line mode changed; a row-per-file read holds the file either way."""
    path = write("a\nb\nc\n")
    batches = list(TextSource(path, mode="file").iter_batches())

    assert sum(b.num_rows for b in batches) == 1
    assert batches[0].column("text")[0].as_py() == "a\nb\nc\n"


def test_read_and_iter_batches_agree(write) -> None:
    path = write(_CASES["mixed_terminators"])
    src = TextSource(path, mode="line")

    from_read = [t for b in src.read() for t in b.column("text").to_pylist()]
    from_iter = [t for b in src.iter_batches() for t in b.column("text").to_pylist()]

    assert from_read == from_iter
