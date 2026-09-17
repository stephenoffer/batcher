"""A binary read that projects only ``uri``/``size`` lists files without reading them.

This is the Batcher spelling of Daft's ``from_glob_path``: ``bt.read.binary(glob)
.select("uri", "size")``. It used to read every payload and then drop it. The spy on
`BinarySource._read_one` is what proves the listing path ran, and the second assertion in each
test is the positive control showing the spy does see a read when payloads are projected.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.formats.unstructured import binary

pytestmark = pytest.mark.io


@pytest.fixture
def corpus(tmp_path):
    for i in range(6):
        (tmp_path / f"f{i}.bin").write_bytes(b"x" * i)
    (tmp_path / "big.bin").write_bytes(b"y" * 3_000_000)
    return tmp_path


@pytest.fixture
def reads(monkeypatch):
    calls: list[str] = []
    original = binary.BinarySource._read_one

    def counting(self, path):
        calls.append(path)
        return original(self, path)

    monkeypatch.setattr(binary.BinarySource, "_read_one", counting)
    return calls


def test_uri_and_size_come_from_the_listing(corpus, reads):
    got = bt.read.binary(str(corpus)).select("uri", "size").sort("uri").to_pydict()
    names = [u.rsplit("/", 1)[-1] for u in got["uri"]]
    assert names == ["big.bin", *(f"f{i}.bin" for i in range(6))]
    assert got["size"] == [3_000_000, 0, 1, 2, 3, 4, 5]
    assert reads == []

    bt.read.binary(str(corpus)).select("uri", "bytes").to_pydict()
    assert len(reads) == 7


def test_a_filter_on_size_still_skips_the_payloads(corpus, reads):
    got = bt.read.binary(str(corpus)).filter(bt.col("size") > 3).select("uri").to_pydict()
    assert len(got["uri"]) == 3
    assert reads == []


def test_projecting_mime_still_reads_because_it_is_sniffed_from_the_bytes(corpus, reads):
    got = bt.read.binary(str(corpus)).select("uri", "mime").to_pydict()
    assert len(got["mime"]) == 7
    assert len(reads) == 7


def test_the_listing_batch_keeps_the_source_types(corpus):
    source = binary.BinarySource(str(corpus))
    batch = next(source.iter_batches(["uri", "size"]))
    assert batch.schema == pa.schema([("uri", pa.string()), ("size", pa.int64())])


def test_an_empty_directory_is_named_rather_than_listed_as_nothing(tmp_path):
    """A directory with no files is an error naming it, not an empty listing."""
    from batcher._internal.errors import BatcherError

    (tmp_path / "empty").mkdir()
    with pytest.raises(BatcherError, match="empty"):
        bt.read.binary(str(tmp_path / "empty")).select("uri", "size").to_pydict()


def test_daft_from_glob_path_lists_the_same_files_and_sizes(corpus):
    daft = pytest.importorskip("daft")
    ours = bt.read.binary(str(corpus)).select("uri", "size").to_pydict()
    theirs = daft.from_glob_path(f"{corpus}/*").to_pydict()
    strip = lambda u: u.removeprefix("file://")  # noqa: E731
    assert sorted(zip(ours["uri"], ours["size"], strict=True)) == sorted(
        zip(map(strip, theirs["path"]), theirs["size"], strict=True)
    )
