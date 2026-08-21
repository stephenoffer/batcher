"""`ds.write(..., mode=...)` save-mode semantics (Spark `SaveMode` parity).

``overwrite`` (default) replaces; ``error`` raises if the path exists; ``ignore``
skips if it exists; ``append`` is only valid for the lakehouse sinks.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_overwrite_is_default(tmp_path):
    p = str(tmp_path / "out.parquet")
    _ds().write.parquet(p)
    m = _ds().write.parquet(p)  # overwrites without error
    assert m.num_files == 1
    assert bt.read.parquet(p).count() == 3


def test_error_mode_raises_when_exists(tmp_path):
    p = str(tmp_path / "out.parquet")
    _ds().write.parquet(p)
    with pytest.raises(PlanError, match="already exists"):
        _ds().write(p, mode="error")


def test_error_mode_writes_when_absent(tmp_path):
    p = str(tmp_path / "fresh.parquet")
    m = _ds().write(p, mode="error")
    assert m.num_files == 1


def test_ignore_mode_skips_when_exists(tmp_path):
    p = str(tmp_path / "out.parquet")
    _ds().write.parquet(p)
    # Overwrite the file's content would change it; ignore must leave it untouched.
    m = bt.from_pydict({"x": [99], "y": [99]}).write(p, mode="ignore")
    assert m.num_files == 0
    assert bt.read.parquet(p).count() == 3  # original rows preserved


def test_append_unsupported_on_file_format(tmp_path):
    p = str(tmp_path / "out.parquet")
    with pytest.raises(PlanError, match="append"):
        _ds().write(p, mode="append")


def test_unknown_mode_raises(tmp_path):
    """An unrecognized save mode is refused, and the message says what is accepted.

    The regex used to be "unknown mode", which the error has not said for as long as it has
    listed the accepted modes -- so this test failed at HEAD. What is asserted now is the
    part a caller needs: the rejected value is quoted back, and the alternatives are there
    to pick from.
    """
    p = str(tmp_path / "out.parquet")
    with pytest.raises(PlanError, match="Unknown save mode") as refused:
        _ds().write(p, mode="upsert")
    message = str(refused.value)
    assert "'upsert'" in message, "the message must quote the mode it rejected"
    for canonical in ("overwrite", "error", "ignore", "append"):
        assert canonical in message, f"{canonical} is a canonical mode and must be offered"
