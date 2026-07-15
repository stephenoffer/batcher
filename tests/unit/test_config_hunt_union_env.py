"""Bug-hunt regression: env overrides of a ``bool | str`` union config field.

The defect: ``_scalar_type`` leaves a two-member union (``bool | str``, as on
``distributed.runtime_bloom_join = "auto"``) unresolved, so ``_coerce`` returned the raw
*string* uncoerced. ``BATCHER_DISTRIBUTED_RUNTIME_BLOOM_JOIN=true`` then shipped the
string ``"true"``, which failed validation (``must be True, False, or 'auto'``) — while the
string sentinel ``"auto"`` happened to pass. So a user could set the string form but a
boolean env value crashed with ``ConfigError``. The fix coerces a recognized boolean token
to a real bool while leaving a string sentinel alone.
"""

from __future__ import annotations

import pytest

from batcher.config.config import Config


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("on", True),
        ("yes", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("off", False),
        ("no", False),
        ("auto", "auto"),  # the string sentinel still passes through unchanged
    ],
)
def test_runtime_bloom_join_env_coercion(raw: str, expected: object) -> None:
    cfg = Config.from_env({"BATCHER_DISTRIBUTED_RUNTIME_BLOOM_JOIN": raw})
    assert cfg.distributed.runtime_bloom_join == expected
    # And the type is exactly right (a real bool, not the string "true"/"false").
    assert type(cfg.distributed.runtime_bloom_join) is type(expected)
