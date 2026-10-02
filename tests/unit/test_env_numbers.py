"""A malformed numeric `BATCHER_*` knob warns and falls back rather than breaking import.

The IO knobs are parsed into module constants at import time. A strict `int(...)` there made
one typo (`BATCHER_REMOTE_READ_CONCURRENCY=32x`) fail `import batcher.io` with a bare
`ValueError` naming neither the variable nor its default.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

from batcher.config.env import env_float, env_int

pytestmark = pytest.mark.unit


def test_env_int_parses_floors_and_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("BATCHER_IO_THREADS", " 12 ")
    assert env_int("BATCHER_IO_THREADS", 64) == 12
    assert env_int("BATCHER_IO_THREADS", 64, floor=16) == 16
    monkeypatch.setenv("BATCHER_IO_THREADS", "")
    assert env_int("BATCHER_IO_THREADS", 64) == 64
    monkeypatch.setenv("BATCHER_IO_THREADS", "32x")
    with caplog.at_level(logging.WARNING, logger="batcher.config.env"):
        assert env_int("BATCHER_IO_THREADS", 64) == 64
    assert "BATCHER_IO_THREADS='32x'" in caplog.text


def test_env_float_falls_back(monkeypatch):
    monkeypatch.setenv("BATCHER_READ_RETRY_BACKOFF_S", "fast")
    assert env_float("BATCHER_READ_RETRY_BACKOFF_S", 0.5) == 0.5
    monkeypatch.setenv("BATCHER_READ_RETRY_BACKOFF_S", "2.5")
    assert env_float("BATCHER_READ_RETRY_BACKOFF_S", 0.5) == 2.5


def test_a_malformed_io_knob_does_not_break_import():
    env = {**os.environ, "BATCHER_REMOTE_READ_CONCURRENCY": "32x", "BATCHER_ORC_STRIPE_BYTES": "8M"}
    code = (
        "import batcher.io.base.source as s, batcher.io.formats.structured.orc as o; "
        "print(s._REMOTE_READ_CONCURRENCY, o._STRIPE_BYTES)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["32", str(8 << 20)]
