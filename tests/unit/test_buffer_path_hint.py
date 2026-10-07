"""Passing file content where a path belongs names the pyarrow idiom (AP-425)."""

from __future__ import annotations

import io

import pytest

import batcher as bt
from batcher._internal.errors import IOError as BtIOError


@pytest.mark.parametrize(
    "buf",
    [io.BytesIO(b"a\n1\n"), io.StringIO("a\n1\n"), bytearray(b"a\n1\n"), b"a\n1\n"],
    ids=["BytesIO", "StringIO", "bytearray", "bytes"],
)
def test_a_buffer_names_from_arrow(buf):
    with pytest.raises(BtIOError, match="from_arrow"):
        bt.read.csv(buf)


def test_the_named_idiom_works():
    import pyarrow.csv

    assert bt.from_arrow(pyarrow.csv.read_csv(io.BytesIO(b"a\n1\n2\n"))).count() == 2
