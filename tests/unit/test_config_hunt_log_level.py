"""Bug-hunt regression: ``_level_value`` fallback on an unknown log-level name.

The defect: ``_level_value`` documented "defaulting to WARNING on an unknown name" but
returned ``logging.getLevelName(name.upper())`` directly. For an unrecognized name that
call returns the *string* ``"Level FOO"`` rather than an int, and ``logger.setLevel`` on a
string it does not recognize raises ``ValueError`` — so ``configure`` with an unvalidated
``ObservabilityConfig`` crashed instead of falling back. The fix guards on the return type.
"""

from __future__ import annotations

import logging

import pytest

from batcher._internal.logging import _level_value, configure
from batcher.config import ObservabilityConfig


@pytest.mark.unit
def test_level_value_unknown_name_falls_back_to_warning() -> None:
    assert _level_value("BOGUS") == logging.WARNING
    assert isinstance(_level_value("BOGUS"), int)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),  # case-insensitive
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_level_value_known_names(name: str, expected: int) -> None:
    assert _level_value(name) == expected


@pytest.mark.unit
def test_configure_with_unknown_level_does_not_raise() -> None:
    # Directly configuring with an out-of-enum level (bypassing config validation) must
    # not raise — it falls back to WARNING rather than a ValueError from setLevel.
    configure(ObservabilityConfig(log_level="BOGUS", console=False))
    assert logging.getLogger("batcher").level == logging.WARNING
