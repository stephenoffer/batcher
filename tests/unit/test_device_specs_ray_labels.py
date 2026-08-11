"""The device table must recognize the names Ray actually labels nodes with.

`ray.util.accelerators` exposes constants whose *identifiers* (`NVIDIA_TESLA_T4`) look like
this table's row keys and whose *values* (`"T4"`) are what a node's `ray.io/accelerator-type`
label carries. Only the value ever reaches `device_spec`, so a table that matched identifiers
alone reported "unknown" for every NVIDIA datacenter part on every real cluster — silently,
because unknown is a legitimate answer that every caller degrades gracefully to.

These tests pin the mapping against the live `ray.util.accelerators` module, so a part Ray adds
later fails here rather than rejoining the unknown path unnoticed.
"""

from __future__ import annotations

import pytest

from batcher._internal.device_specs import device_spec, known_device_names, resolve_device_name
from batcher._internal.device_specs.table import RAY_LABEL_ALIASES, SPECS

pytestmark = pytest.mark.unit

#: Accelerators Ray can label a node with that this table deliberately does not carry. Each is
#: absent because no row was written for it, not because its name failed to match — so the
#: right behavior is `None` ("unknown"), and mapping one onto a near neighbour would be exactly
#: the fabricated figure the table exists to avoid.
_KNOWN_ABSENT = frozenset(
    {
        "AMD-Radeon-HD-7900",
        "AMD-Radeon-R9-200-HD-7900",
        "Intel-GAUDI",
        "MXC500",
        "MXC550",
        "aws-neuron-core",
    }
)


def _ray_label_values() -> dict[str, str]:
    """Every `ray.util.accelerators` constant, as `{identifier: label value}`.

    Selected by "module-level, not private, string-valued" rather than by `name.isupper()`.
    Ray does not spell every constant in strict upper case — `AMD_INSTINCT_MI300x` carries a
    lowercase tail — and an `isupper()` filter silently dropped exactly those from the sweep.
    That is how `AMD-Instinct-MI300X-OAM`, the only spelling a real MI300X node is labelled
    with, went unresolved past a green test: the part this file exists to catch.
    """
    accelerators = pytest.importorskip("ray.util.accelerators")
    return {
        name: value
        for name, value in vars(accelerators).items()
        if not name.startswith("_") and isinstance(value, str) and not name.startswith("__")
    }


def test_every_ray_label_either_resolves_or_is_known_absent() -> None:
    """No Ray-labelled accelerator falls through to unknown by accident."""
    unresolved = {
        value for value in _ray_label_values().values() if resolve_device_name(value) is None
    }
    assert unresolved <= _KNOWN_ABSENT, (
        f"Ray labels these accelerators but the device table does not recognize them: "
        f"{sorted(unresolved - _KNOWN_ABSENT)}. Add a row to `table._ROWS`, or an entry to "
        f"`RAY_LABEL_ALIASES` if the part is already there under a different spelling."
    )


def test_known_absent_list_has_no_stale_entries() -> None:
    """A part that gains a row must leave the absent list, or the list stops meaning anything."""
    labels = set(_ray_label_values().values())
    resolvable_but_listed = {
        value for value in _KNOWN_ABSENT & labels if resolve_device_name(value) is not None
    }
    assert not resolvable_but_listed, (
        f"these are listed as absent but now resolve: {sorted(resolvable_but_listed)} — "
        f"drop them from _KNOWN_ABSENT"
    )


def test_resolved_labels_carry_real_facts() -> None:
    """Resolving is only worth anything if the row behind it is populated."""
    for identifier, label in _ray_label_values().items():
        if label in _KNOWN_ABSENT:
            continue
        spec = device_spec(label)
        assert spec is not None, f"{identifier}={label!r} resolves but has no spec"
        assert spec.memory_gib > 0, f"{identifier}={label!r} resolved to a row with no VRAM"


def test_bare_part_names_reach_the_right_row() -> None:
    """The spellings this cluster's fleet is labelled with, pinned individually."""
    assert device_spec("T4") is SPECS["NVIDIA_TESLA_T4"]
    assert device_spec("A100") is SPECS["NVIDIA_A100"]
    assert device_spec("H100") is SPECS["NVIDIA_H100"]
    assert device_spec("A10G") is SPECS["NVIDIA_A10G"]
    assert device_spec("L4") is SPECS["NVIDIA_L4"]
    assert device_spec("V100") is SPECS["NVIDIA_TESLA_V100"]


def test_bare_a100_takes_the_smaller_variant() -> None:
    """A label with no memory size sizes for the card that would OOM, not the one that fits."""
    assert device_spec("A100").memory_gib == 40
    assert device_spec("A100-80G").memory_gib == 80


def test_alias_targets_all_exist() -> None:
    """Every alias points at a real row."""
    missing = {alias: key for alias, key in RAY_LABEL_ALIASES.items() if key not in SPECS}
    assert not missing, f"aliases pointing at rows that do not exist: {missing}"


def test_aliases_do_not_shadow_a_real_row() -> None:
    """An alias must never redirect a name the table already carries."""
    shadowed = sorted(set(RAY_LABEL_ALIASES) & set(SPECS))
    assert not shadowed, f"aliases shadow real row keys: {shadowed}"


def test_row_keys_still_resolve_to_themselves() -> None:
    """The identifier spelling keeps working — the alias layer is additive."""
    for key in known_device_names():
        assert resolve_device_name(key) == key


def test_punctuation_spelling_is_accepted() -> None:
    """A label writer's choice of separator is not a device fact."""
    assert device_spec("AMD-Instinct-MI300X") is SPECS["AMD_INSTINCT_MI300X"]
    assert device_spec("amd_instinct_mi300x") is SPECS["AMD_INSTINCT_MI300X"]


def test_unknown_still_reads_unknown() -> None:
    """The alias layer must not turn a genuinely unrecognized part into a guess."""
    assert device_spec("NVIDIA_MADE_UP_9000") is None
    assert resolve_device_name("wildly-unknown-part") is None
    assert device_spec(None) is None
    assert device_spec("") is None
