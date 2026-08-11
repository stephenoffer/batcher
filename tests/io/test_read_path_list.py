"""A list of paths accepts the same vocabulary one path does.

``read.parquet([a, b])`` is the shape `pandas.concat` and ``spark.read.parquet(*paths)``
cover, and the entries are not always files: pointing it at two output *directories* is
the natural way to union two runs. That used to fail with "path is a directory", raised
from inside the format's per-file reader -- an error about the wrong thing, from the wrong
layer.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


@pytest.fixture
def two_runs(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    bt.from_pydict({"v": [1, 2]}).write.parquet(a, max_rows_per_file=1)
    bt.from_pydict({"v": [3, 4]}).write.parquet(b, max_rows_per_file=1)
    return a, b


def _values(ds) -> list[int]:
    return sorted(ds.to_pydict()["v"])


def test_a_list_of_directories_reads_as_one_relation(two_runs):
    assert _values(bt.read.parquet(list(two_runs))) == [1, 2, 3, 4]


def test_a_list_of_globs_reads_as_one_relation(two_runs):
    a, b = two_runs
    assert _values(bt.read.parquet([f"{a}/*.parquet", f"{b}/*.parquet"])) == [1, 2, 3, 4]


def test_a_list_of_files_still_reads_exactly_those(two_runs):
    a, _ = two_runs
    files = sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith(".parquet"))
    assert _values(bt.read.parquet(files)) == [1, 2]


def test_a_file_named_twice_is_read_once(two_runs):
    a, _ = two_runs
    one = sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith(".parquet"))[0]
    # The directory already covers the file, so naming both must not double its rows.
    assert _values(bt.read.parquet([a, one])) == [1, 2]


def test_a_list_of_directories_works_through_the_generic_read(two_runs):
    assert _values(bt.read(list(two_runs), format="parquet")) == [1, 2, 3, 4]


def test_a_list_of_csv_directories_reads_as_one_relation(tmp_path):
    a, b = str(tmp_path / "ca"), str(tmp_path / "cb")
    bt.from_pydict({"v": [1, 2]}).write.csv(a, max_rows_per_file=1)
    bt.from_pydict({"v": [3, 4]}).write.csv(b, max_rows_per_file=1)
    assert _values(bt.read.csv([a, b])) == [1, 2, 3, 4]
