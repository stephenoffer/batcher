"""AMD accelerators are visible without ROCm, and an unrepairable HBM error condemns a board.

Every device fact Batcher reads elsewhere comes from NVML, which is NVIDIA-only, so on an
Instinct node the telemetry, fault, and mode probes all reported nothing and that read as a
healthy idle host. `hardware.amdgpu` reads the `amdgpu` driver's own sysfs tree instead, which
needs no ROCm install and no device context.

These build a fake `/sys/class/drm` in a tmp directory, because the real one on this machine
has no AMD card in it and a test that only runs on the hardware it describes never runs. The
patch target is the implementation module, not the package facade: rebinding the constant on
the facade would leave the reader inside `devices` still looking at the real `/sys`.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.amd import devices as amdgpu

pytestmark = pytest.mark.unit


def _card(root, number, *, vendor="0x1002", ras=None, hwmon=None, **attrs):
    """Write one fake DRM card, returning its `device` directory."""
    device = root / f"card{number}" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(f"{vendor}\n")
    for name, value in attrs.items():
        (device / name).write_text(f"{value}\n")
    if hwmon is not None:
        sensors = device / "hwmon" / "hwmon3"
        sensors.mkdir(parents=True)
        for name, value in hwmon.items():
            (sensors / name).write_text(f"{value}\n")
    if ras is not None:
        block_dir = device / "ras"
        block_dir.mkdir()
        for block, (correctable, uncorrectable) in ras.items():
            (block_dir / f"{block}_err_count").write_text(
                f"ue: {uncorrectable}\nce: {correctable}\n"
            )
    return device


@pytest.fixture
def drm(tmp_path, monkeypatch):
    """A fake DRM tree, with the probe's memo cleared on the way in and out."""
    root = tmp_path / "drm"
    root.mkdir()
    monkeypatch.setattr(amdgpu, "AMDGPU_SYSFS_ROOT", str(root))
    amdgpu.reset_amd_probe()
    yield root
    amdgpu.reset_amd_probe()


# --- Reading a device -------------------------------------------------------------------------


def test_a_full_instinct_board_reads_every_figure(drm):
    _card(
        drm,
        0,
        product_name="AMD Instinct MI300X",
        unique_id="a1b2c3d4e5f60718",
        serial_number="PCB0123456",
        mem_info_vram_total=str(192 * (1 << 30)),
        mem_info_vram_used=str(40 * (1 << 30)),
        gpu_busy_percent="87",
        hwmon={
            "temp1_input": "71000",
            "temp1_crit": "100000",
            "power1_average": "540000000",
            "power1_cap": "750000000",
        },
        ras={"umc": (12, 0), "gfx": (0, 0)},
    )
    (device,) = amdgpu.amd_devices()
    assert device.name == "AMD Instinct MI300X"
    assert device.serial_number == "PCB0123456"
    assert device.memory_total_bytes == 192 * (1 << 30)
    assert device.memory_free_bytes == 152 * (1 << 30)
    assert device.busy_percent == 87
    # Millidegrees and microwatts are the hwmon contract; a caller wants degrees and watts.
    assert device.temperature_c == 71.0
    assert device.temperature_limit_c == 100.0
    assert device.power_watts == 540.0
    assert device.power_cap_watts == 750.0
    assert device.thermal_headroom_c == 29.0
    assert device.uncorrectable_errors == 0
    assert {block.block for block in device.ras} == {"umc", "gfx"}


def test_the_board_name_resolves_into_the_device_table(drm):
    # The point of reading `product_name` at all: it is the string the spec table is keyed on,
    # so an AMD node gets its real bandwidth and TDP rather than an unknown-device default.
    from batcher._internal.device_specs import device_spec, resolve_device_name

    _card(drm, 0, product_name="AMD Instinct MI300X", mem_info_vram_total="1")
    (device,) = amdgpu.amd_devices()
    spec = device_spec(resolve_device_name(device.name))
    assert spec is not None and spec.vendor == "amd"


def test_a_missing_attribute_is_unknown_not_a_crash(drm):
    # The realistic shape: an older kernel publishes the memory figures and nothing else.
    _card(drm, 0, mem_info_vram_total=str(64 * (1 << 30)))
    (device,) = amdgpu.amd_devices()
    assert device.memory_total_bytes == 64 * (1 << 30)
    assert (device.name, device.serial_number, device.ras) == ("", "", ())
    assert (device.temperature_c, device.power_watts, device.busy_percent) == (0.0, 0.0, 0)


def test_a_garbled_attribute_is_unknown_too(drm):
    _card(drm, 0, mem_info_vram_total="not a number", gpu_busy_percent="")
    (device,) = amdgpu.amd_devices()
    assert (device.memory_total_bytes, device.busy_percent) == (0, 0)


# --- Which devices count ----------------------------------------------------------------------


def test_a_non_amd_card_is_not_an_amd_device(drm):
    # `/sys/class/drm` carries every vendor's display adapter. Without the vendor check a
    # laptop's Intel iGPU would report as a datacenter accelerator with unknown everything.
    _card(drm, 0, vendor="0x8086", mem_info_vram_total="1")
    _card(drm, 1, vendor="0x10de", mem_info_vram_total="1")
    assert amdgpu.amd_devices() == ()
    assert amdgpu.amd_present() is False


def test_a_display_connector_is_not_a_device(drm):
    # `card0-DP-1` is a connector node sitting beside `card0` and has no `device/vendor`.
    _card(drm, 0, mem_info_vram_total="1")
    (drm / "card0-DP-1").mkdir()
    (drm / "card0-eDP-1").mkdir()
    assert len(amdgpu.amd_devices()) == 1


def test_devices_are_numbered_densely_over_the_amd_cards(drm):
    # A host with a display adapter at card0 and Instincts after it must number those from 0,
    # because that is how ROCm and every operator tool number them. Numbering by DRM node
    # would offset every index by one and silently mis-pair a fault with a device.
    _card(drm, 0, vendor="0x1a03", mem_info_vram_total="1")
    _card(drm, 1, product_name="a", mem_info_vram_total="1")
    _card(drm, 2, product_name="b", mem_info_vram_total="1")
    assert [d.index for d in amdgpu.amd_devices()] == [0, 1]
    assert [d.card for d in amdgpu.amd_devices()] == ["card1", "card2"]


def test_ten_or_more_cards_stay_in_numeric_order(drm):
    # Lexical ordering puts card10 between card1 and card2, which reorders half a dense node.
    for number in range(12):
        _card(drm, number, product_name=f"gpu{number}", mem_info_vram_total="1")
    assert [d.card for d in amdgpu.amd_devices()] == [f"card{n}" for n in range(12)]


# --- Faults -------------------------------------------------------------------------------------


def test_an_unrepairable_memory_error_condemns_the_board(drm):
    _card(drm, 0, product_name="mi300x", ras={"umc": (4, 2)})
    (faulted,) = amdgpu.ecc_faulted_amd_devices()
    assert faulted.memory_uncorrectable_errors == 2
    assert faulted.uncorrectable_errors == 2


def test_a_recovered_engine_fault_does_not_condemn_the_board(drm):
    # An uncorrectable error in a compute block can come from one bad command and clears on a
    # reset. Treating it like a failed HBM would take a healthy board out of the fleet.
    _card(drm, 0, ras={"gfx": (0, 1), "sdma": (0, 3), "umc": (99, 0)})
    (device,) = amdgpu.amd_devices()
    assert device.uncorrectable_errors == 4
    assert device.memory_uncorrectable_errors == 0
    assert amdgpu.ecc_faulted_amd_devices() == ()


def test_correctable_errors_alone_are_not_a_fault(drm):
    _card(drm, 0, ras={"umc": (100_000, 0)})
    assert amdgpu.ecc_faulted_amd_devices() == ()


def test_an_absent_ras_tree_reports_no_faults_rather_than_zero_faults(drm):
    _card(drm, 0, mem_info_vram_total="1")
    (device,) = amdgpu.amd_devices()
    assert device.ras == ()
    assert amdgpu.ecc_faulted_amd_devices() == ()


# --- Throttling ---------------------------------------------------------------------------------


def test_a_board_at_its_power_cap_is_reported_as_clock_limited(drm):
    _card(drm, 0, hwmon={"power1_average": "749000000", "power1_cap": "750000000"})
    (limited,) = amdgpu.throttled_amd_devices()
    assert limited.power_headroom < 0.02


def test_a_board_against_its_critical_temperature_is_reported(drm):
    _card(drm, 0, hwmon={"temp1_input": "98000", "temp1_crit": "100000"})
    assert len(amdgpu.throttled_amd_devices()) == 1


def test_a_cool_board_with_headroom_is_not_throttled(drm):
    _card(
        drm,
        0,
        hwmon={
            "temp1_input": "60000",
            "temp1_crit": "100000",
            "power1_average": "300000000",
            "power1_cap": "750000000",
        },
    )
    assert amdgpu.throttled_amd_devices() == ()


def test_a_board_that_publishes_no_limits_is_never_called_throttled(drm):
    # Unknown is not throttled. Comparing an unreadable draw against an unreadable cap is how
    # every device on a node that cannot read hwmon gets flagged at once.
    _card(drm, 0, product_name="mi210")
    (device,) = amdgpu.amd_devices()
    assert (device.power_headroom, device.thermal_headroom_c) == (1.0, 0.0)
    assert amdgpu.throttled_amd_devices() == ()


def test_the_threshold_is_the_board_s_own_limit_not_a_constant(drm):
    # Two parts at the same 85 C, with different critical points. Judging both against one
    # hard-coded number calls one of them wrong whichever number is chosen.
    _card(drm, 0, hwmon={"temp1_input": "85000", "temp1_crit": "87000"})
    _card(drm, 1, hwmon={"temp1_input": "85000", "temp1_crit": "110000"})
    limited = amdgpu.throttled_amd_devices()
    assert [d.index for d in limited] == [0]


# --- Visibility ---------------------------------------------------------------------------------


def test_no_drm_tree_reports_unreadable_rather_than_healthy(tmp_path, monkeypatch):
    monkeypatch.setattr(amdgpu, "AMDGPU_SYSFS_ROOT", str(tmp_path / "absent"))
    amdgpu.reset_amd_probe()
    assert amdgpu.readable() is False
    assert amdgpu.amd_devices() == ()
    assert amdgpu.ecc_faulted_amd_devices() == ()
    amdgpu.reset_amd_probe()


def test_an_nvidia_only_host_is_unreadable_for_amd_purposes(drm):
    _card(drm, 0, vendor="0x10de", mem_info_vram_total="1")
    assert amdgpu.readable() is False, "a tree with no AMD card in it is not AMD visibility"


def test_a_populated_tree_is_readable(drm):
    _card(drm, 0, product_name="mi300x")
    assert amdgpu.readable() is True


def test_this_host_answers_without_raising():
    # Against the real machine, whatever it is. The contract is that every entry point is safe
    # to call anywhere, which is what lets the report call them unconditionally.
    amdgpu.reset_amd_probe()
    assert isinstance(amdgpu.amd_present(), bool)
    assert isinstance(amdgpu.amd_devices(), tuple)
    assert isinstance(amdgpu.throttled_amd_devices(), tuple)


# --- Reaching the report ------------------------------------------------------------------------


def test_an_amd_node_enumerates_its_devices(drm, monkeypatch):
    # The inventory is what every other layer asks "what is attached". Before this it returned
    # `[]` on an Instinct node with no ROCm torch, so the whole GPU path read as absent.
    from batcher._internal import accelerators

    _card(drm, 0, product_name="AMD Instinct MI300X", mem_info_vram_total=str(192 * (1 << 30)))
    monkeypatch.setattr(accelerators, "_nvml_inventory", lambda: [])
    accelerators._gpu_inventory_probe.cache_clear()
    try:
        inventory = accelerators.gpu_inventory()
    finally:
        accelerators._gpu_inventory_probe.cache_clear()
    assert inventory == [
        {"index": 0, "name": "AMD Instinct MI300X", "memory_bytes": 192 * (1 << 30)}
    ]


def test_an_amd_fault_reaches_the_problem_list_through_the_shared_row_keys(drm, monkeypatch):
    # The wiring that matters: `accelerator_problems` knows nothing about AMD, and does not
    # need to, because the AMD reading lands in the keys the NVIDIA path already publishes.
    from batcher._internal import accelerators
    from batcher.api.session.accelerators.report import accelerator_problems

    _card(
        drm,
        0,
        product_name="AMD Instinct MI300X",
        mem_info_vram_total=str(192 * (1 << 30)),
        ras={"umc": (3, 1)},
        hwmon={"temp1_input": "99000", "temp1_crit": "100000"},
    )
    monkeypatch.setattr(accelerators, "_nvml_inventory", lambda: [])
    accelerators._gpu_inventory_probe.cache_clear()
    try:
        problems = accelerator_problems()
    finally:
        accelerators._gpu_inventory_probe.cache_clear()
    assert any("unrepairable HBM error" in p for p in problems), problems
    assert any("critical limit" in p for p in problems), problems


# --- Reaching the admission decision --------------------------------------------------------


def test_a_failed_hbm_takes_the_device_out_of_the_fleet(drm):
    from batcher.carbonite.accel import amd_verdicts

    _card(drm, 0, product_name="mi300x", unique_id="dead01", ras={"umc": (0, 1)})
    (verdict,) = amd_verdicts()
    assert verdict.state == "quarantine"
    assert verdict.schedulable is False
    assert verdict.reasons == ("hbm_uncorrectable",)
    assert verdict.uuid == "dead01", "health history is keyed on the device's own id"


def test_a_recovered_engine_fault_derates_rather_than_quarantines(drm):
    from batcher.carbonite.accel import amd_verdicts

    _card(drm, 0, ras={"gfx": (0, 2)})
    (verdict,) = amd_verdicts()
    assert verdict.state == "degraded"
    assert verdict.schedulable is True
    assert verdict.reasons == ("engine_uncorrectable",)


def test_a_hot_board_is_judged_against_its_own_critical_point(drm):
    # The same rule the NVIDIA path applies to a published slowdown point: the lower of the
    # board's limit and the configured ceiling. A board that clamps at 90 must not be judged
    # by a fleet-wide 87 it would trip long before its own hardware cares.
    from batcher.carbonite.accel import HealthThresholds, amd_verdicts

    _card(drm, 0, hwmon={"temp1_input": "84000", "temp1_crit": "88000"})
    (verdict,) = amd_verdicts(HealthThresholds(max_temperature_c=95.0))
    assert "hot" in verdict.reasons, "88 C limit less the 5 C margin is 83, and it is at 84"


def test_a_healthy_board_produces_a_healthy_verdict(drm):
    from batcher.carbonite.accel import amd_verdicts

    _card(
        drm,
        0,
        product_name="mi300x",
        mem_info_vram_total=str(192 * (1 << 30)),
        mem_info_vram_used=str(20 * (1 << 30)),
        hwmon={"temp1_input": "62000", "temp1_crit": "100000"},
        ras={"umc": (0, 0)},
    )
    (verdict,) = amd_verdicts()
    assert (verdict.state, verdict.reasons, verdict.derate) == ("healthy", (), 1.0)


def test_a_full_board_is_not_admitted_onto(drm):
    from batcher.carbonite.accel import amd_verdicts

    _card(drm, 0, mem_info_vram_total="100", mem_info_vram_used="99")
    (verdict,) = amd_verdicts()
    assert "memory_full" in verdict.reasons


def test_the_live_fleet_assessment_falls_through_to_amd_when_nvml_is_silent(drm, monkeypatch):
    # The wiring: `assess_fleet()` is what the scheduler and the fleet probe call, and on an
    # Instinct node it returned an empty tuple that every caller read as "nothing to worry
    # about" rather than "nothing was looked at".
    from batcher.carbonite.accel import health

    _card(drm, 0, product_name="mi300x", ras={"umc": (0, 5)})
    monkeypatch.setattr(
        "batcher._internal.hardware.nvml.device_telemetry", lambda: (), raising=True
    )
    verdicts = health.assess_fleet()
    assert [v.state for v in verdicts] == ["quarantine"]


def test_passing_no_readings_deliberately_still_means_no_verdicts(drm, monkeypatch):
    # `assess_fleet(())` is how a caller asks to judge nothing. The AMD fall-through must not
    # turn that into a fleet-wide probe on a path that explicitly opted out.
    from batcher.carbonite.accel import health

    _card(drm, 0, ras={"umc": (0, 5)})
    assert health.assess_fleet((), None, ()) == ()


def test_an_amd_fault_reaches_the_node_condition_gauge(drm):
    # One gauge for both vendors: a fleet does not want two alerts for "a device's memory has
    # failed", and NVML reports nothing at all on the AMD half of a mixed fleet.
    from batcher.observe.metrics import _node_conditions

    _card(drm, 0, ras={"umc": (0, 2)})
    assert _node_conditions()["faulted_devices"] >= 1
