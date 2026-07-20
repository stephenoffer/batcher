"""Blob sources batch by bytes as well as by file count.

Media and binary file sizes span orders of magnitude inside one directory, so a batch
bounded only by a file count is unbounded in memory — 64 thumbnails is 256 KB, 64 videos
is 12.8 GB. These tests pin the packing arithmetic (an exact cover, both bounds honored)
and, just as importantly, that `iter_batches`, `read` and `splits` all cut the corpus in
the *same* places: a split that chunked differently would make a distributed read return a
different set of batches from the single-node one, which is invariant #7.
"""

from __future__ import annotations

import random

import pytest

from batcher.io.formats.multimodal._batching import pack_by_count_and_bytes, probe_sizes
from batcher.io.formats.unstructured.binary import BinarySource

pytestmark = pytest.mark.unit

_PNG_HEADER = bytes.fromhex("89504e470d0a1a0a")


# ---- the packing arithmetic -------------------------------------------------


def test_packs_by_bytes_when_the_byte_bound_binds_first() -> None:
    assert pack_by_count_and_bytes(["a", "b", "c"], [10, 10, 10], 8, 25) == [["a", "b"], ["c"]]


def test_packs_by_count_when_the_count_bound_binds_first() -> None:
    assert pack_by_count_and_bytes(["a", "b", "c"], [1, 1, 1], 2, 1000) == [["a", "b"], ["c"]]


def test_a_file_larger_than_the_budget_gets_its_own_group() -> None:
    """It must still be readable — dropping it or merging it are both wrong."""
    assert pack_by_count_and_bytes(["big", "x"], [500, 1], 8, 100) == [["big"], ["x"]]


def test_empty_input() -> None:
    assert pack_by_count_and_bytes([], [], 8, 100) == []


def test_packing_is_always_an_exact_ordered_cover() -> None:
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(0, 40)
        files = [f"f{i}" for i in range(n)]
        sizes = [rng.randint(0, 300) for _ in range(n)]
        groups = pack_by_count_and_bytes(
            files, sizes, rng.randint(1, 8), rng.randint(1, 400)
        )
        assert [f for g in groups for f in g] == files, "not an exact, ordered cover"
        assert all(g for g in groups), "emitted an empty group"


def test_probe_sizes_treats_an_unstattable_file_as_zero() -> None:
    """A metadata hiccup must not fail the read; it just falls back to the count bound."""

    def size_of(path: str) -> int:
        if path == "bad":
            raise OSError("no such file")
        return 7

    assert probe_sizes(["a", "bad", "b"], size_of) == [7, 0, 7]


# ---- the source wiring ------------------------------------------------------


@pytest.fixture
def mixed_corpus(tmp_path):
    """Three tiny files and one that dwarfs them — the shape that skews a fixed count."""
    for i in range(3):
        (tmp_path / f"s{i}.bin").write_bytes(b"x" * 100)
    (tmp_path / "zbig.bin").write_bytes(b"y" * (3 << 20))
    return str(tmp_path)


def test_a_huge_file_is_isolated_from_the_small_ones(mixed_corpus) -> None:
    src = BinarySource(mixed_corpus, batch_bytes=1 << 20)
    assert [b.num_rows for b in src.iter_batches()] == [3, 1]


def test_read_and_iter_batches_agree(mixed_corpus) -> None:
    src = BinarySource(mixed_corpus, batch_bytes=1 << 20)
    assert [b.num_rows for b in src.read()] == [b.num_rows for b in src.iter_batches()]


def test_splits_cut_where_iter_batches_cuts(mixed_corpus) -> None:
    """Invariant #7: a distributed read must produce the same batches as a local one."""
    src = BinarySource(mixed_corpus, batch_bytes=1 << 20)
    per_split = [sum(b.num_rows for b in s.read()) for s in src.splits()]

    assert per_split == [b.num_rows for b in src.iter_batches()]


def test_splits_honor_an_explicit_target_size(mixed_corpus) -> None:
    """`target_size` used to be accepted and ignored."""
    src = BinarySource(mixed_corpus, batch_bytes=1 << 20)

    assert len(src.splits()) == 2
    assert len(src.splits(target_size=1 << 30)) == 1


def test_the_byte_bound_does_not_disturb_a_uniform_corpus(tmp_path) -> None:
    """Small files must still batch by count, exactly as before."""
    for i in range(10):
        (tmp_path / f"f{i}.bin").write_bytes(b"z" * 10)
    src = BinarySource(str(tmp_path), batch_files=4)

    assert [b.num_rows for b in src.iter_batches()] == [4, 4, 2]


def test_blob_column_uses_64_bit_offsets(tmp_path) -> None:
    """32-bit `binary` overflows at 2 GB *per batch* — reachable with 64 x 32 MB files."""
    import pyarrow as pa

    (tmp_path / "a.bin").write_bytes(b"data")
    src = BinarySource(str(tmp_path))

    assert src.schema().field("bytes").type == pa.large_binary()
    assert src.read()[0].schema.field("bytes").type == pa.large_binary()
