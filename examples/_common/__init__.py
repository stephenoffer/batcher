"""Shared helpers for the example suite: real S3 datasets and device selection.

This package is support code, not an example. The runner
(``tests/docs/test_examples.py``) skips every path with an underscore-prefixed part,
so nothing here is executed as a script.

Scripts reach it with a two-line bootstrap, which works both under the runner and when
you run the file directly:

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from _common import tpch
"""

from __future__ import annotations

from _common.datasets import (
    TPCH_COLUMNS,
    TPCH_ROWS,
    images,
    is_offline,
    tpch,
    tpch_csv_uri,
    tpch_path,
    tpch_uri,
)
from _common.runtime import (
    device_count,
    has_gpu,
    resolve_device,
    resolve_distributed,
    torch_device,
)

__all__ = [
    "TPCH_COLUMNS",
    "TPCH_ROWS",
    "device_count",
    "has_gpu",
    "images",
    "is_offline",
    "resolve_device",
    "resolve_distributed",
    "torch_device",
    "tpch",
    "tpch_csv_uri",
    "tpch_path",
    "tpch_uri",
]
