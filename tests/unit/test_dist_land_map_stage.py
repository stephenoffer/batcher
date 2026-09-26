"""`_land_map_stage` must tell "nothing was written" apart from "the scratch is unreadable".

The staged map stage writes each partition's UDF output to cluster scratch and reads it back
as a Parquet scan. Its caller reads `None` as "every partition was empty" and answers with an
empty table, so deriving `None` from a failed footer read turned a permission error or a
corrupt file into a successful empty result. These tests stub the fan-out (no Ray needed) and
hold the two cases apart.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.dist import executor as dex
from batcher.dist.executors import map as map_mod
from batcher.io.manifest import WriteManifest, WrittenFile

pytestmark = pytest.mark.unit


def _node():
    return bt.from_pydict({"a": [1, 2, 3]})._plan


def _stub_map(monkeypatch, manifest: WriteManifest) -> None:
    monkeypatch.setattr(map_mod, "_distributed_map", lambda *_a, **_k: manifest)


def test_an_unreadable_staged_file_raises_rather_than_reading_as_empty(monkeypatch, tmp_path):
    (tmp_path / "part-0.parquet").write_bytes(b"garbage")
    _stub_map(monkeypatch, WriteManifest((WrittenFile(str(tmp_path / "part-0.parquet"), 3, 7),)))
    with pytest.raises(FormatError):
        dex._land_map_stage(_node(), [], 2, None, str(tmp_path), 1)


def test_a_manifest_with_no_files_is_the_empty_stage(monkeypatch, tmp_path):
    _stub_map(monkeypatch, WriteManifest(()))
    assert dex._land_map_stage(_node(), [], 2, None, str(tmp_path), 1) is None


def test_a_written_stage_is_read_back_as_a_scan(monkeypatch, tmp_path):
    path = tmp_path / "part-0.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), path)
    _stub_map(monkeypatch, WriteManifest((WrittenFile(str(path), 3, path.stat().st_size),)))
    landed = dex._land_map_stage(_node(), [], 2, None, str(tmp_path), 1)
    assert landed is not None
    scan, staged = landed
    assert scan.source_id == 1
    assert staged.schema().names == ["a"]
