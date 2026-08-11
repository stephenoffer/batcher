"""A mistyped write option must fail by name, and a single key must not split.

Both defects here were silent in different ways. A write keyword the sink could not take
surfaced as ``DeltaSink.__init__() got an unexpected keyword argument 'schema_mode'`` — a
class the caller never typed, no suggestion, and on a distributed write it arrived from
inside a Ray worker after the cluster was already up. And ``sort_by="ab"`` was not an
error at all: it unpacked into characters and clustered the output on columns `a` and `b`,
so the zonemaps the option exists to tighten ended up pointing at the wrong keys.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import FormatError


@pytest.fixture
def frame():
    return bt.from_pydict({"id": [1, 2], "v": ["a", "b"]})


@pytest.mark.parametrize(
    ("fmt", "kwargs", "wanted"),
    [
        ("parquet", {"compresion": "zstd"}, "compression"),
        ("orc", {"compresion": "zstd"}, "compression"),
        ("csv", {"delimeter": ";"}, "delimiter"),
    ],
)
def test_a_mistyped_write_option_names_itself_and_suggests_the_real_one(
    tmp_path, frame, fmt, kwargs, wanted
):
    with pytest.raises(FormatError) as excinfo:
        frame.write(str(tmp_path / f"out.{fmt}"), fmt, **kwargs)
    message = str(excinfo.value)
    assert next(iter(kwargs)) in message, "the error must quote what the caller typed"
    assert wanted in message, "and name the option they meant"
    assert "__init__" not in message, "never a constructor the caller cannot see"


def test_a_lakehouse_sink_rejects_by_name_too(tmp_path, frame):
    """Delta/Iceberg are the sinks that leaked the raw `TypeError` this guard replaces."""
    with pytest.raises(FormatError, match="schema_mode"):
        frame.write.delta(str(tmp_path / "t"), mode="overwrite", schema_mode="merge")


@pytest.mark.parametrize(
    ("fmt", "kwargs"),
    [
        ("parquet", {"compression": "snappy"}),
        ("parquet", {"storage_options": {}}),
        ("csv", {"delimiter": ";"}),
        ("csv", {"sep": "|"}),  # an alias the CSV spec knows and a signature would not
        ("csv", {"include_header": False}),
        ("json", {}),
        ("avro", {}),
        ("arrow", {}),
    ],
)
def test_a_real_option_is_not_mistaken_for_a_typo(tmp_path, frame, fmt, kwargs):
    frame.write(str(tmp_path / f"ok.{fmt}"), fmt, **kwargs)


def test_a_csv_option_batcher_deliberately_lacks_still_explains_why(tmp_path, frame):
    """The spec's `unsupported` reason must survive being consulted earlier."""
    with pytest.raises(FormatError, match="not a Batcher option"):
        frame.write.csv(str(tmp_path / "q.csv"), quotechar="'")


def test_a_single_sort_key_is_one_key_and_not_its_letters(tmp_path):
    """`sort_by="ab"` must sort by the column named `ab`, not by columns `a` then `b`.

    The frame is built so the wrong reading succeeds silently: `a` and `b` both exist, so
    unpacking raises nothing and simply clusters on the wrong keys.
    """
    frame = bt.from_pydict({"ab": [3, 1, 2], "a": [9, 9, 9], "b": [3, 2, 1]})
    out = str(tmp_path / "w")
    frame.write.parquet(out, sort_by="ab")
    assert pq.read_table(out).column("ab").to_pylist() == [1, 2, 3]


def test_a_sort_key_list_still_sorts_by_every_key(tmp_path):
    frame = bt.from_pydict({"g": ["y", "x", "x"], "n": [1, 3, 2]})
    out = str(tmp_path / "w")
    frame.write.parquet(out, sort_by=["g", "n"])
    table = pq.read_table(out).to_pydict()
    assert list(zip(table["g"], table["n"], strict=True)) == [("x", 2), ("x", 3), ("y", 1)]
