"""An operator can describe an accelerator the built-in table does not carry.

An unrecognized part reports unknown, and every power, bandwidth and interconnect decision then
keeps the default it had. That is right for a name nobody vouched for and wrong for a part the
operator can describe, so `register_device_spec` is the import path for those figures.
"""

from __future__ import annotations

import pytest

from batcher._internal.device_specs import (
    DeviceSpec,
    accessors,
    device_memory_bandwidth_gbps,
    device_spec,
    device_tdp_watts,
    register_device_spec,
    resolve_device_name,
    table,
)
from batcher._internal.errors import ConfigError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_table():
    specs = dict(table.SPECS)
    aliases = dict(table.RAY_LABEL_ALIASES)
    tokens = accessors._KEY_TOKENS
    yield
    table.SPECS.clear()
    table.SPECS.update(specs)
    table.RAY_LABEL_ALIASES.clear()
    table.RAY_LABEL_ALIASES.update(aliases)
    accessors._KEY_TOKENS = tokens
    resolve_device_name.cache_clear()


# Not "ACME_X9": the GPU-fleet docs page registers that part in its executed example, and
# `tests/docs/test_doc_examples.py` runs the page in the same process -- under xdist, the same
# worker -- so the "unknown until registered" premise was false whenever the page ran first
# (4 failures on gate c5g-gate-m2). A name no page uses keeps the premise this file's own.
_PART = "ACME_UNIT_X9"


def _spec(name: str = "acme-unit-x9", **overrides) -> DeviceSpec:
    fields = {
        "name": name,
        "vendor": "acme",
        "generation": "x",
        "memory_gib": 64,
        "memory_bandwidth_gbps": 2000.0,
        "tdp_watts": 450.0,
        "idle_watts": 60.0,
        "half_tflops": 300.0,
        "fp8_tflops": 0.0,
        "nvlink_domain": 1,
        "nvlink_gbps": 0.0,
        "mig_slices": 0,
    }
    fields.update(overrides)
    return DeviceSpec(**fields)


def test_an_unknown_part_reports_unknown_until_it_is_registered():
    assert device_spec(_PART) is None
    assert device_tdp_watts(_PART) == 0.0
    stored = register_device_spec(_spec(), aliases=("UX9",))
    assert stored.name == _PART
    assert device_tdp_watts(_PART) == 450.0
    assert device_memory_bandwidth_gbps("UX9") == 2000.0
    # The driver-name resolver sees the new row too, not only exact lookups.
    assert resolve_device_name("Acme Unit X9 64GB") == _PART


def test_a_measured_row_replaces_the_nameplate_one():
    before = device_spec("NVIDIA_H100")
    assert before is not None
    register_device_spec(_spec("NVIDIA_H100", memory_bandwidth_gbps=2900.0))
    assert device_memory_bandwidth_gbps("NVIDIA_H100") == 2900.0


@pytest.mark.parametrize("bad", [{"name": "--"}, {"tdp_watts": -1.0}, {"memory_gib": -8}])
def test_an_invalid_row_is_refused(bad):
    with pytest.raises(ConfigError):
        register_device_spec(_spec(**bad))
    assert device_spec(_PART) is None
