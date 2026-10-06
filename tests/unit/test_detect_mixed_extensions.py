"""A path list that mixes formats is refused by name, not as a parse failure (AP-426).

Inference reads the format off the first entry, so ``[x.parquet, y.csv]`` used to fail as
"could not read 'y.csv' as parquet", blaming the file's bytes for a mistake in the list.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import FormatError
from batcher.io.detect import detect_format


def test_mixed_formats_name_both_entries():
    with pytest.raises(FormatError, match="mixes formats") as err:
        detect_format(["data/x.parquet", "data/y.csv"])
    assert "x.parquet" in str(err.value) and "y.csv" in str(err.value)


def test_one_format_passes():
    assert detect_format(["a.parquet", "b.parquet"]) == "parquet"


def test_compression_suffix_is_the_same_format():
    assert detect_format(["a.csv", "b.csv.gz"]) == "csv"


def test_an_explicit_format_is_not_second_guessed():
    assert detect_format(["a.txt", "b.csv"], explicit="csv") == "csv"


def test_entries_without_an_extension_keep_the_first_entry_rule():
    assert detect_format(["a.parquet", "some/dir/"]) == "parquet"
