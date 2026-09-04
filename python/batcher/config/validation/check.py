"""The one-line refusal primitive every section check is written in terms of.

Its own module because `sections.py` and `distributed.py` both need it, and the four
subsystems' rule against copy-paste applies just as well inside a package: a second
`_check` would be a second place for the exception type to drift.
"""

from __future__ import annotations

from batcher._internal.errors import ConfigError

__all__ = ["check"]


def check(cond: bool, msg: str) -> None:
    """Raise `ConfigError(msg)` unless `cond` holds.

    Args:
        cond: The condition that must hold for the value to be accepted.
        msg: The message naming the field, its bound, and the value seen.

    Raises:
        ConfigError: When `cond` is false.
    """
    if not cond:
        raise ConfigError(msg)
