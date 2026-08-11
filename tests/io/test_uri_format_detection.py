"""Format detection reads the extension off the last path segment.

Truncating the path at the first ``*`` threw the extension away with it, so *every* glob
failed detection: ``bt.read("data/*.parquet")`` raised `FormatError` even though it is the
documented spelling for reading many files, and so did
``s3://bucket/*.parquet?endpoint_override=...``, which is how the filesystem docs say to
address an on-prem S3. Neither needs to touch storage to be answerable -- the suffix is
right there in the name.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.detect import detect_format

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("uri", "fmt"),
    [
        ("data/*.parquet", "parquet"),
        ("s3://bucket/*.parquet", "parquet"),
        ("s3://bucket/part-*.parquet", "parquet"),
        ("s3://bucket/*/*.parquet", "parquet"),
        ("/tmp/nowhere/*.csv", "csv"),
        ("gs://b/logs/*.jsonl", "json"),
        # A query string carries connection config; it is not part of the name.
        ("s3://b/k.parquet?endpoint_override=https://minio:9000", "parquet"),
        ("s3://b/*.parquet?endpoint_override=https://minio:9000", "parquet"),
        # Compression still wraps rather than replaces the format.
        ("events.csv.gz", "csv"),
        ("s3://b/events.csv.gz?region=eu", "csv"),
        ("s3://b/*.csv.gz", "csv"),
        # Plain paths are unaffected.
        ("/a/b.parquet", "parquet"),
        ("delta://table", "delta"),
    ],
)
def test_the_extension_is_read_off_the_last_segment(uri, fmt):
    assert detect_format(uri) == fmt


@pytest.mark.parametrize("uri", ["data/2024-*/", "s3://bucket/dir/", "s3://bucket/*"])
def test_a_path_whose_last_segment_has_no_extension_still_asks(uri):
    # A wildcard *directory* names no format, and guessing one would be inventing it.
    with pytest.raises(FormatError):
        detect_format(uri)


def test_a_glob_read_works_end_to_end_without_naming_the_format(tmp_path):
    bt.from_pydict({"v": [1, 2, 3]}).write.parquet(str(tmp_path / "out"), max_rows_per_file=1)
    assert sorted(bt.read(f"{tmp_path}/out/*.parquet").to_pydict()["v"]) == [1, 2, 3]


def test_a_glob_read_of_csv_works_too(tmp_path):
    bt.from_pydict({"v": [1, 2]}).write.csv(str(tmp_path / "out"), max_rows_per_file=1)
    assert sorted(bt.read(f"{tmp_path}/out/*.csv").to_pydict()["v"]) == [1, 2]
