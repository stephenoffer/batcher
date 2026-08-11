"""Both RFC 8089 spellings of a ``file:`` URI must name the same local path.

``file:///tmp/x`` carries an empty authority and was accepted; ``file:/tmp/x`` omits the
authority and was not, failing from inside pyarrow with ``Expected a local filesystem
path, got a URI``. The second form is the one ``java.net.URI`` normalizes to, so it is
what Hadoop's `Path`, a Spark listing, and Java-stack manifests print — exactly the paths
someone migrating pastes in.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.filesystem import local_path


@pytest.fixture
def table_dir(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    bt.from_pydict({"n": [1, 2, 3]}).write.parquet(str(directory / "a.parquet"))
    return str(directory)


@pytest.mark.parametrize("scheme", ["", "file://", "file:"])
@pytest.mark.parametrize("shape", ["dir", "file", "glob"])
def test_every_file_uri_spelling_reads_the_same_rows(table_dir, scheme, shape):
    target = {"dir": table_dir, "file": f"{table_dir}/a.parquet", "glob": f"{table_dir}/*.parquet"}
    assert bt.read.parquet(scheme + target[shape]).count() == 3


@pytest.mark.parametrize("scheme", ["", "file://", "file:"])
def test_every_file_uri_spelling_is_writable(tmp_path, scheme):
    out = f"{scheme}{tmp_path}/out-{scheme.count('/')}"
    bt.from_pydict({"n": [7]}).write.parquet(out)
    assert bt.read.parquet(out).to_pydict()["n"] == [7]


@pytest.mark.parametrize(
    ("given", "wanted"),
    [
        ("file:///tmp/x", "/tmp/x"),
        ("file:/tmp/x", "/tmp/x"),
        ("/tmp/x", "/tmp/x"),
        ("s3://bucket/k", "s3://bucket/k"),
        # A relative name that merely contains a colon is a filename, not a URI: the
        # prefix is claimed only when a '/' follows the scheme.
        ("file:relative", "file:relative"),
    ],
)
def test_local_path_strips_only_a_real_file_scheme(given, wanted):
    assert local_path(given) == wanted
