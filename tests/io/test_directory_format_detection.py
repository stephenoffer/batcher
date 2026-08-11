"""`read(dir)` infers its format from the files in the directory.

A directory has no extension of its own, so reading one back used to demand
``format="parquet"`` for a layout the *writer* chose -- the least guessable argument in
the API, since the caller never named a format on the way in. Detection now looks inside,
and declines (rather than guessing) whenever the answer is not unambiguous.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt
from batcher._internal.errors import FormatError

pytestmark = pytest.mark.integration


def test_a_directory_of_parquet_parts_needs_no_format(tmp_path):
    out = str(tmp_path / "parts")
    bt.from_pydict({"v": list(range(10))}).write.parquet(out, max_rows_per_file=4)
    assert sorted(bt.read(out).to_pydict()["v"]) == list(range(10))


def test_a_hive_directory_needs_no_format_and_keeps_its_partition_column(tmp_path):
    out = str(tmp_path / "hive")
    bt.from_pydict({"g": ["a", "b"], "v": [1, 2]}).write.parquet(out, partition_by=["g"])
    assert sorted(bt.read(out).columns) == ["g", "v"]


def test_a_directory_of_csv_parts_needs_no_format(tmp_path):
    out = str(tmp_path / "csvparts")
    bt.from_pydict({"v": list(range(9))}).write.csv(out, max_rows_per_file=4)
    assert sorted(bt.read(out).to_pydict()["v"]) == list(range(9))


def test_a_mixed_directory_declines_rather_than_picking_one(tmp_path):
    # Two formats under one root is not one relation. Guessing would read half the data.
    out = tmp_path / "mixed"
    out.mkdir()
    (out / "a.parquet").write_bytes(b"")
    (out / "b.csv").write_text("v\n1\n")
    with pytest.raises(FormatError):
        bt.read(str(out))


def test_an_empty_directory_still_asks_for_the_format(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(FormatError):
        bt.read(str(out))


def test_an_explicit_format_still_wins(tmp_path):
    out = str(tmp_path / "explicit")
    bt.from_pydict({"v": [1, 2, 3]}).write.parquet(out, max_rows_per_file=2)
    assert bt.read(out, format="parquet").count() == 3


def test_a_missing_path_reports_the_format_it_could_not_infer(tmp_path):
    with pytest.raises(FormatError):
        bt.read(os.path.join(str(tmp_path), "nothing-here"))


def test_a_named_file_is_unaffected(tmp_path):
    path = str(tmp_path / "one.parquet")
    bt.from_pydict({"x": [1, 2]}).write.parquet(path)
    assert bt.read(path).to_pydict() == {"x": [1, 2]}
