"""The plain-text and XML writers, and the text reader's blank-line filter.

Spark's `DataFrameWriter.text` and `.xml` had no Batcher counterpart. Both writers are held to
the same contract as every other sink: what they write reads back with the right types, nulls
and empty input keep a defined shape, a large value survives, and a streaming write encodes
one batch at a time instead of collecting the result first (that last property is pinned for
every new sink at once in `test_sinks_stream_batch_by_batch.py`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError, SchemaError
from batcher.io.formats.base import SINKS
from batcher.io.formats.semistructured.xml import XMLSink
from batcher.io.formats.unstructured.text import TextSink

pytestmark = pytest.mark.io


def test_both_sinks_are_registered_under_their_format_names():
    assert SINKS.get("text") is TextSink
    assert SINKS.get("xml") is XMLSink


# --- text -----------------------------------------------------------------------------------


def test_text_round_trips_values_and_writes_a_null_as_an_empty_line(tmp_path):
    out = str(tmp_path / "lines.txt")
    manifest = bt.from_pydict({"value": ["alpha", None, "", "gamma"]}).write.text(out)
    assert manifest.files[0].rows == 4
    assert (tmp_path / "lines.txt").read_bytes() == b"alpha\n\n\ngamma\n"

    back = bt.read.text(out)
    assert back.schema.field("text").type == pa.string()
    assert back.to_pydict()["text"] == ["alpha", "", "", "gamma"]
    assert back.to_pydict()["line_number"] == [1, 2, 3, 4]


def test_text_writes_an_empty_file_for_an_empty_result(tmp_path):
    out = str(tmp_path / "empty.txt")
    ds = bt.from_pydict({"value": ["a", "b"]}).filter(bt.col("value") == "zzz")
    ds.write.text(out)
    assert (tmp_path / "empty.txt").read_bytes() == b""
    assert bt.read.text(out).count() == 0


def test_text_keeps_a_large_value_and_a_custom_separator(tmp_path):
    big = "x" * 3_000_000
    out = str(tmp_path / "big.txt")
    bt.from_pydict({"value": [big, "tail"]}).write.text(out, line_sep="\r\n")
    raw = (tmp_path / "big.txt").read_bytes()
    assert raw == big.encode() + b"\r\ntail\r\n"
    assert bt.read.text(out).to_pydict()["text"] == [big, "tail"]


def test_text_directory_write_splits_rows_across_part_files(tmp_path):
    out = str(tmp_path / "parts")
    values = [f"row-{i}" for i in range(50)]
    manifest = bt.from_pydict({"value": values}).write.text(out, max_rows_per_file=20)
    assert [f.rows for f in manifest.files] == [20, 20, 10]
    assert sorted(bt.read.text(out).to_pydict()["text"]) == sorted(values)


@pytest.mark.parametrize(
    "data",
    [{"n": [1, 2]}, {"a": ["x"], "b": ["y"]}, {"nested": [{"k": "v"}]}],
    ids=["not-a-string", "two-columns", "a-struct"],
)
def test_text_refuses_anything_but_one_string_column(tmp_path, data):
    with pytest.raises(SchemaError, match="exactly one string column"):
        bt.from_pydict(data).write.text(str(tmp_path / "bad.txt"))


def test_text_refuses_an_empty_separator():
    with pytest.raises(FormatError, match="line_sep"):
        TextSink(line_sep="")


def test_text_output_reads_back_the_same_in_daft(tmp_path):
    daft = pytest.importorskip("daft")
    out = str(tmp_path / "d.txt")
    values = ["one", "", "  ", "four"]
    bt.from_pydict({"value": values}).write.text(out)
    assert daft.read_text(out, skip_blank_lines=False).to_pydict()["text"] == values
    ours = bt.read.text(out, skip_blank_lines=True).select("text").to_pydict()["text"]
    assert ours == daft.read_text(out).to_pydict()["text"] == ["one", "four"]


# --- read.text(skip_blank_lines=) -------------------------------------------------------------


def test_skip_blank_lines_drops_empty_and_whitespace_lines_and_keeps_line_numbers(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("a\n\n   \n\tb\n\t\n")
    kept = bt.read.text(str(path), skip_blank_lines=True).to_pydict()
    assert kept["text"] == ["a", "\tb"]
    assert kept["line_number"] == [1, 4]
    # Positive control: without the flag every line is there.
    assert bt.read.text(str(path)).count() == 5


def test_skip_blank_lines_is_refused_in_file_mode(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a\n")
    with pytest.raises(FormatError, match="mode='line'"):
        bt.read.text(str(path), mode="file", skip_blank_lines=True)


# --- xml ------------------------------------------------------------------------------------


def _rows(path, row_tag="ROW"):
    return ET.parse(path).getroot().findall(row_tag)


def test_xml_writes_spark_layout_with_nested_lists_attributes_and_nulls(tmp_path):
    import datetime

    out = str(tmp_path / "books.xml")
    ds = bt.from_pydict(
        {
            "_id": ["b1", "b2"],
            "title": ["Dune & <Sons>", None],
            "tags": [["sf", "classic"], []],
            "author": [{"_lang": "en", "name": "Frank"}, None],
            "published": [datetime.date(1965, 8, 1), None],
            "cover": [b"\x00\xff", None],
            "in_print": [True, False],
            "price": [9.5, None],
        }
    )
    manifest = ds.write.xml(out, row_tag="book", root_tag="books")
    assert manifest.files[0].rows == 2
    text = (tmp_path / "books.xml").read_text()
    assert text.startswith('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<books>')

    first, second = _rows(out, "book")
    assert first.get("id") == "b1"
    assert first.findtext("title") == "Dune & <Sons>"
    assert [t.text for t in first.findall("tags")] == ["sf", "classic"]
    assert first.find("author").get("lang") == "en"
    assert first.find("author").findtext("name") == "Frank"
    assert first.findtext("published") == "1965-08-01"
    assert first.findtext("cover") == "AP8="
    assert first.findtext("in_print") == "true"
    assert first.findtext("price") == "9.5"
    # Nulls and an empty list write no element at all.
    assert [child.tag for child in second] == ["in_print"]
    assert second.get("id") == "b2"


def test_xml_null_value_writes_an_element_for_a_null(tmp_path):
    out = str(tmp_path / "n.xml")
    bt.from_pydict({"a": [None]}).write.xml(out, null_value="NULL", declaration="")
    assert (tmp_path / "n.xml").read_text() == "<ROWS>\n<ROW><a>NULL</a></ROW>\n</ROWS>\n"


def test_xml_empty_result_is_a_valid_document_with_no_rows(tmp_path):
    out = str(tmp_path / "e.xml")
    bt.from_pydict({"a": [1]}).filter(bt.col("a") > 5).write.xml(out)
    assert _rows(out) == []


def test_xml_keeps_a_large_value(tmp_path):
    big = "é" * 1_000_000
    out = str(tmp_path / "big.xml")
    bt.from_pydict({"body": [big]}).write.xml(out)
    assert _rows(out)[0].findtext("body") == big


def test_xml_refuses_a_map_and_an_unusable_element_name(tmp_path):
    table = pa.table({"m": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64()))})
    with pytest.raises(SchemaError, match="map"):
        bt.from_arrow(table).write.xml(str(tmp_path / "m.xml"))
    with pytest.raises(FormatError, match="valid XML element name"):
        bt.from_pydict({"bad name": [1]}).write.xml(str(tmp_path / "b.xml"))
    with pytest.raises(FormatError, match="row_tag"):
        XMLSink(row_tag="1row")
