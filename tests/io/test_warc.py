"""The WARC source: record framing, header typing, gzip members, and splits.

A WARC has no index, so the framing *is* the format: a record's payload length comes from
its `Content-Length` header and nothing else. These tests build records by hand rather than
through a library, because a library would agree with itself about the framing and the
point is to check the reader against the spec's bytes.
"""

from __future__ import annotations

import datetime as dt
import gzip
import pickle
from pathlib import Path

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.formats.unstructured.warc import WARC_SCHEMA, WarcSource


def record(
    *,
    rtype: str = "response",
    uri: str = "https://example.com/",
    body: bytes = b"<html>hi</html>",
    date: str = "2024-03-15T13:45:30Z",
    extra: str = "",
    record_id: str = "urn:uuid:11111111-1111-1111-1111-111111111111",
) -> bytes:
    """One WARC record, built from the spec's bytes rather than through a library."""
    headers = (
        f"WARC/1.0\r\n"
        f"WARC-Type: {rtype}\r\n"
        f"WARC-Record-ID: <{record_id}>\r\n"
        f"WARC-Date: {date}\r\n"
        f"WARC-Target-URI: {uri}\r\n"
        f"Content-Type: application/http; msgtype=response\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{extra}"
        f"\r\n"
    ).encode()
    return headers + body + b"\r\n\r\n"


@pytest.fixture
def crawl(tmp_path: Path) -> Path:
    """A two-record WARC: one response with an extension header, one request."""
    path = tmp_path / "crawl.warc"
    path.write_bytes(
        record(extra="WARC-IP-Address: 1.2.3.4\r\n")
        + record(rtype="request", uri="https://example.com/a", body=b"GET / HTTP/1.1")
    )
    return path


def test_reads_one_row_per_record(crawl):
    out = bt.read.warc(str(crawl)).collect()
    assert out.num_rows == 2
    assert out.schema == WARC_SCHEMA
    assert out.column("warc_type").to_pylist() == ["response", "request"]
    assert out.column("warc_content").to_pylist() == [b"<html>hi</html>", b"GET / HTTP/1.1"]


def test_named_headers_become_typed_columns(crawl):
    out = bt.read.warc(str(crawl)).collect()
    assert out.column("warc_date").to_pylist()[0] == dt.datetime(2024, 3, 15, 13, 45, 30)
    assert out.column("warc_content_length").to_pylist() == [15, 14]
    assert out.column("warc_target_uri").to_pylist()[1] == "https://example.com/a"
    assert out.column("warc_record_id").to_pylist()[0].startswith("<urn:uuid:")


def test_unnamed_headers_survive_as_json(crawl):
    # The point of the JSON tail: a crawl's own extension headers are what a provenance or
    # dedup pass needs, and this source cannot know their names.
    out = bt.read.warc(str(crawl)).select(
        ip=bt.col("warc_headers").json.extract_string("$.warc-ip-address")
    )
    assert out.to_pydict()["ip"] == ["1.2.3.4", None]


def test_headers_column_is_always_an_object_never_null(crawl):
    # So a `.json` accessor on it needs no null guard.
    out = bt.read.warc(str(crawl)).select(t=bt.col("warc_headers").json.type_of())
    assert out.to_pydict()["t"] == ["object", "object"]


def test_payload_containing_the_record_delimiter_is_read_whole(tmp_path):
    # The reason the reader trusts Content-Length rather than scanning for `\r\n\r\n`: an
    # HTTP response body contains that sequence as a matter of course.
    body = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html>body</html>"
    path = tmp_path / "c.warc"
    path.write_bytes(record(body=body) + record(body=b"second"))
    out = bt.read.warc(str(path)).collect()
    assert out.num_rows == 2
    assert out.column("warc_content").to_pylist() == [body, b"second"]


def test_gzip_members_are_read_transparently(tmp_path):
    # A `.warc.gz` is conventionally per-record gzip members concatenated, not one member
    # over the file — that is what lets a reader start at a record boundary.
    path = tmp_path / "c.warc.gz"
    path.write_bytes(gzip.compress(record()) + gzip.compress(record(rtype="request")))
    out = bt.read.warc(str(path)).collect()
    assert out.column("warc_type").to_pylist() == ["response", "request"]


def test_a_single_gzip_member_over_the_whole_file_also_works(tmp_path):
    path = tmp_path / "whole.warc.gz"
    path.write_bytes(gzip.compress(record() + record(rtype="request")))
    assert bt.read.warc(str(path)).collect().num_rows == 2


def test_format_is_detected_from_the_extension(tmp_path):
    for name in ("c.warc", "c2.warc.gz"):
        path = tmp_path / name
        data = record()
        path.write_bytes(gzip.compress(data) if name.endswith(".gz") else data)
        assert bt.read(str(path)).collect().num_rows == 1, name


def test_folded_header_lines_are_joined(tmp_path):
    # RFC 822 continuation: a header may wrap onto an indented line.
    path = tmp_path / "c.warc"
    path.write_bytes(record(extra="WARC-Long: first\r\n   second\r\n"))
    out = bt.read.warc(str(path)).select(
        v=bt.col("warc_headers").json.extract_string("$.warc-long")
    )
    assert out.to_pydict()["v"] == ["first second"]


def test_an_unparseable_date_is_null_rather_than_a_failed_read(tmp_path):
    path = tmp_path / "c.warc"
    path.write_bytes(record(date="not a date"))
    out = bt.read.warc(str(path)).collect()
    assert out.column("warc_date").to_pylist() == [None]
    assert out.column("warc_type").to_pylist() == ["response"]


def test_a_truncated_record_raises_by_default(tmp_path):
    path = tmp_path / "c.warc"
    path.write_bytes(record()[:-10])  # payload shorter than Content-Length promises
    with pytest.raises(FormatError, match="truncated"):
        bt.read.warc(str(path)).collect()


def test_a_truncated_record_is_tolerated_under_skip(tmp_path):
    # A WARC is a stream: once framing is wrong the reader's position is wrong, so the
    # file stops contributing rather than emitting garbage after the bad record.
    path = tmp_path / "c.warc"
    path.write_bytes(record() + record(rtype="request")[:-10])
    source = WarcSource(str(path), on_error="skip")
    batches = source.read(None)
    assert sum(b.num_rows for b in batches) == 1
    assert source.corrupt_files()


def test_a_non_warc_file_raises(tmp_path):
    path = tmp_path / "c.warc"
    path.write_bytes(b"this is not a warc file at all\r\n\r\n")
    with pytest.raises(FormatError, match="not a WARC record"):
        bt.read.warc(str(path)).collect()


def test_empty_file_reads_as_no_rows(tmp_path):
    path = tmp_path / "c.warc"
    path.write_bytes(b"")
    assert bt.read.warc(str(path)).collect().num_rows == 0


def test_projection_is_honored(crawl):
    out = bt.read.warc(str(crawl)).select("warc_type", "warc_target_uri").collect()
    assert out.column_names == ["warc_type", "warc_target_uri"]
    assert out.num_rows == 2


def test_splits_cover_the_source_exactly_once_and_survive_pickling(tmp_path):
    for i in range(3):
        (tmp_path / f"c{i}.warc").write_bytes(record(uri=f"https://example.com/{i}"))
    source = WarcSource(str(tmp_path))
    whole = source.read(None)
    whole_uris = [
        u
        for b in whole
        for u in b.column(WARC_SCHEMA.get_field_index("warc_target_uri")).to_pylist()
    ]

    split_uris: list[str] = []
    for split in source.splits():
        revived = pickle.loads(pickle.dumps(split))  # the distributed path in miniature
        for batch in revived.read(None):
            idx = WARC_SCHEMA.get_field_index("warc_target_uri")
            split_uris += batch.column(idx).to_pylist()
    assert sorted(split_uris) == sorted(whole_uris)
    assert len(split_uris) == 3


def test_iter_batches_and_read_agree(crawl):
    source = WarcSource(str(crawl))
    assert pa.Table.from_batches(source.read(None), WARC_SCHEMA) == pa.Table.from_batches(
        list(source.iter_batches(None)), WARC_SCHEMA
    )


def test_registered_under_its_name():
    from batcher.io.formats.base import SOURCES

    assert "warc" in SOURCES.names()
    assert SOURCES.get("warc") is WarcSource
