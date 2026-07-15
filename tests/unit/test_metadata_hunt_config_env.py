"""Bug-hunt regression: env overrides of Optional-typed numeric config fields.

The defect: ``_overlay_env`` coerced an env var against ``type(current)``. For an
``int | None`` / ``float | None`` field whose default is ``None`` that runtime type is
``NoneType``, so ``_coerce`` skipped coercion and stored the raw *string*. The string
then either crashed validation (``str > int`` TypeError) or shipped a wrong-typed value
across the Rust engine-config wire contract. The fix resolves the field's *declared*
scalar type from the annotation.
"""

from __future__ import annotations

import pytest

from batcher.config.config import Config


@pytest.mark.parametrize(
    ("env_key", "section", "field", "raw", "expected"),
    [
        ("BATCHER_MEMORY_MAX_MEMORY_BYTES", "memory", "max_memory_bytes", "1073741824", 1073741824),
        (
            "BATCHER_MEMORY_SPILL_LOCAL_BUDGET_BYTES",
            "memory",
            "spill_local_budget_bytes",
            "500",
            500,
        ),
        (
            "BATCHER_DISTRIBUTED_FLIGHT_KEEPALIVE_S",
            "distributed",
            "flight_keepalive_s",
            "30",
            30.0,
        ),
    ],
)
def test_env_override_optional_numeric_field(env_key, section, field, raw, expected) -> None:
    cfg = Config.from_env({env_key: raw})
    value = getattr(getattr(cfg, section), field)
    assert value == expected
    assert type(value) is type(expected)  # int stays int, float stays float — never str


def test_env_override_optional_str_field_stays_string() -> None:
    cfg = Config.from_env({"BATCHER_MEMORY_SPILL_DIR": "/tmp/spill"})
    assert cfg.memory.spill_dir == "/tmp/spill"


def test_env_override_regular_fields_unaffected() -> None:
    cfg = Config.from_env(
        {"BATCHER_EXECUTION_MORSEL_ROWS": "4096", "BATCHER_EXECUTION_FUSE_LINEAR": "0"}
    )
    assert cfg.execution.morsel_rows == 4096
    assert cfg.execution.fuse_linear is False
