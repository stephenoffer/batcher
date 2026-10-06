"""A JSON-array document read by default gets the array-document error (AP-430).

It used to surface as ``ArrowInvalid: JSON parse error: Column() changed from object to
array in row 0``, which names neither the file's shape nor the conversion that fixes it.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import FormatError


@pytest.mark.parametrize(
    "text", ['[{"a":1},{"a":2}]', '  \n[\n  {"a": 1},\n  {"a": 2}\n]\n'], ids=["compact", "pretty"]
)
def test_an_array_document_names_the_conversion(tmp_path, text):
    path = tmp_path / "arr.json"
    path.write_text(text)
    with pytest.raises(FormatError, match="JSON-array file") as err:
        bt.read.json(str(path)).to_pydict()
    assert "orient='records', lines=True" in str(err.value)
    assert "ArrowInvalid" not in str(err.value)


def test_newline_delimited_json_is_unaffected(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text('{"a":1}\n{"a":2}\n')
    assert bt.read.json(str(path)).to_pydict() == {"a": [1, 2]}


def test_a_malformed_object_file_keeps_its_own_error(tmp_path):
    # The control: only a file that *opens* with '[' is called an array document.
    path = tmp_path / "bad.json"
    path.write_text('{"a":1}\n{"a":\n')
    with pytest.raises(FormatError) as err:
        bt.read.json(str(path)).to_pydict()
    assert "JSON-array file" not in str(err.value)
