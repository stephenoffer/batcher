"""Reading a directory must find data files nested below it.

`expand` listed a directory with `recursive=False` and raised "no <suffix> files found" when
nothing sat at the top level. That made two normal layouts unreadable: a Hive tree Batcher
itself wrote with `partition_by=`, and a media corpus laid out `videos/2024/01/...` — the
standard shape for a large multimodal training corpus.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _write_nested(root, rel: str, rows: list[int]) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    bt.from_pydict({"x": rows}).write.parquet(str(target))


def test_a_hive_partitioned_directory_batcher_wrote_reads_back(tmp_path):
    out = str(tmp_path / "hive")
    bt.from_pydict({"x": [1, 2, 3], "p": ["a", "a", "b"]}).write.parquet(out, partition_by=["p"])
    got = bt.read(out, format="parquet").collect().to_pydict()
    assert sorted(got["x"]) == [1, 2, 3]


def test_a_deeply_nested_corpus_is_discovered(tmp_path):
    root = tmp_path / "corpus"
    _write_nested(root, "2024/01/a.parquet", [1, 2])
    _write_nested(root, "2024/02/b.parquet", [3])
    _write_nested(root, "2025/03/c.parquet", [4, 5])

    got = bt.read(str(root), format="parquet").collect().to_pydict()
    assert sorted(got["x"]) == [1, 2, 3, 4, 5]


def test_a_flat_directory_still_reads_without_descending(tmp_path):
    """The fast path is unchanged: a flat layout must not start walking subdirectories."""
    root = tmp_path / "flat"
    root.mkdir()
    bt.from_pydict({"x": [1, 2]}).write.parquet(str(root / "a.parquet"))
    bt.from_pydict({"x": [3]}).write.parquet(str(root / "b.parquet"))
    # A subdirectory of decoys: if the flat listing matched, these are never visited.
    (root / "nested").mkdir()
    bt.from_pydict({"x": [99]}).write.parquet(str(root / "nested" / "decoy.parquet"))

    got = bt.read(str(root), format="parquet").collect().to_pydict()
    assert sorted(got["x"]) == [1, 2, 3]


def test_an_empty_directory_still_raises_an_actionable_error(tmp_path):
    root = tmp_path / "empty"
    (root / "sub").mkdir(parents=True)
    with pytest.raises(Exception, match=r"no \.parquet files found"):
        bt.read(str(root), format="parquet").collect()
