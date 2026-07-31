"""The kernel log as a fault source: record timestamps, age windows, and node-level faults.

Two properties are load-bearing here and neither is obvious from the code.

The first is that **evidence expires**. The ring buffer holds a node's history, so an Xid from
before the last device reset is still sitting in it. A quarantine keyed on "the buffer contains
a fatal code" therefore never releases a device that has since been repaired, and a fleet
shrinks monotonically over its lifetime with nothing in any log to explain it. Every windowed
read below exists to stop that.

The second is that **an undated record is not an old record**. A log without the kernel's own
header — a `dmesg` dump, a journal export, a fixture — carries no timestamp, and aging those
out would silently drop live faults on exactly the deployments least able to see them.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.faults import actions, kernel, kmsg, xid

pytestmark = pytest.mark.unit


def _log(tmp_path, lines: list[str], name: str = "kmsg") -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _record(seconds: float, message: str) -> str:
    """A record in the kernel's documented `<prio>,<seq>,<usec>,<flag>;<text>` shape."""
    return f"6,1,{int(seconds * 1_000_000)},-;{message}"


# --- the ring-buffer reader ---------------------------------------------------------------


def test_a_record_header_yields_the_kernel_timestamp(tmp_path):
    path = _log(tmp_path, [_record(12.5, "NVRM: something happened")])
    (record,) = kmsg.read_kmsg(path)
    assert record.timestamp_s == pytest.approx(12.5)
    assert record.text == "NVRM: something happened"


def test_a_header_without_the_flag_field_still_yields_a_timestamp(tmp_path):
    # `dmesg --raw` and several journal exports print three header fields, not four. The
    # timestamp is the field that matters and it is present in both shapes.
    path = _log(tmp_path, ["6,1,7500000;NVRM: three-field header"])
    (record,) = kmsg.read_kmsg(path)
    assert record.timestamp_s == pytest.approx(7.5)


def test_a_line_with_no_header_is_kept_with_an_unknown_timestamp(tmp_path):
    path = _log(tmp_path, ["NVRM: a plain dmesg dump has no header"])
    (record,) = kmsg.read_kmsg(path)
    assert record.timestamp_s == -1.0
    assert record.age_s() == -1.0
    assert record.text.startswith("NVRM:")


def test_continuation_lines_are_not_records(tmp_path):
    # `/dev/kmsg` follows each record with indented `KEY=value` metadata. Counting those as
    # records would let one kernel message match a fault pattern more than once.
    path = _log(
        tmp_path,
        [_record(1.0, "Out of memory: Killed process 42 (python)"), " SUBSYSTEM=mm", " DEVICE=+mm"],
    )
    assert len(kmsg.read_kmsg(path)) == 1


def test_an_unreadable_log_reads_as_empty_and_says_so(tmp_path):
    missing = str(tmp_path / "absent")
    assert kmsg.read_kmsg(missing) == ()
    assert kmsg.kmsg_readable(missing) is False
    assert kmsg.kmsg_readable(_log(tmp_path, [_record(1.0, "hello")])) is True


def test_the_record_ceiling_bounds_a_pathological_log(tmp_path):
    path = _log(tmp_path, [_record(i, f"line {i}") for i in range(50)])
    assert len(kmsg.read_kmsg(path, max_records=10)) <= 10


# --- Xid windowing ------------------------------------------------------------------------


#: What the boot clock reads in these tests. Pinned rather than sampled, because a record's
#: age is measured against the *real* uptime, and a host that booted ten minutes ago cannot
#: hold a record from yesterday — so a window test written against the live clock passes or
#: fails depending on how long the machine has been up.
_NOW_S = 500_000.0


@pytest.fixture
def frozen_clock(monkeypatch):
    """Pin the boot clock both fault readers age records against."""
    monkeypatch.setattr(kmsg, "monotonic_now_s", lambda: _NOW_S)
    monkeypatch.setattr(xid, "monotonic_now_s", lambda: _NOW_S)
    monkeypatch.setattr(kernel, "monotonic_now_s", lambda: _NOW_S)


def _stale_and_fresh(tmp_path) -> str:
    """A log holding one Xid from a day ago and one from just now, on the pinned clock."""
    return _log(
        tmp_path,
        [
            _record(_NOW_S - 86_400, "NVRM: Xid (PCI:0000:0c:00): 94, contained ECC"),
            _record(_NOW_S, "NVRM: Xid (PCI:0000:1a:00): 95, uncontained ECC"),
        ],
    )


def test_a_stale_xid_ages_out_of_the_window(tmp_path, frozen_clock):
    path = _stale_and_fresh(tmp_path)
    fresh = xid.recent_xid_events(path, within_s=3600.0)
    assert [e.code for e in fresh] == [95]
    # Without a window the whole buffer is still available, which is what a forensic read wants.
    assert [e.code for e in xid.recent_xid_events(path)] == [94, 95]


def test_the_quarantine_map_defaults_to_a_window_not_to_all_of_history(tmp_path, frozen_clock):
    # This is the property that lets a reset device come back: a device whose only fatal code
    # is a day old is not in the map a scheduler acts on, because the default window is hours.
    events = xid.recent_xid_events(_stale_and_fresh(tmp_path), within_s=xid.XID_WINDOW_S)
    assert xid.xid_fatal(events) == {"0000:1a:00.0": (95,)}


def test_an_undated_xid_survives_every_window(tmp_path):
    path = _log(tmp_path, ["NVRM: Xid (PCI:0000:0c:00): 79, fell off the bus"])
    assert [e.code for e in xid.recent_xid_events(path, within_s=1.0)] == [79]


def test_repeat_counts_separate_one_fault_from_a_storm(tmp_path):
    path = _log(
        tmp_path,
        [_record(1.0 + i, "NVRM: Xid (PCI:0000:0c:00): 31, MMU fault") for i in range(5)]
        + [_record(9.0, "NVRM: Xid (PCI:0000:1a:00): 31, MMU fault")],
    )
    counts = xid.xid_counts(xid.recent_xid_events(path))
    assert counts[("0000:0c:00.0", 31)] == 5
    assert counts[("0000:1a:00.0", 31)] == 1


# --- remedies and trust -------------------------------------------------------------------


def test_every_classified_code_has_a_documented_remedy():
    # A code the fault path quarantines on but has no repair for leaves an operator with a
    # dead slot and no ticket to open, which is how a fleet shrinks silently.
    assert actions.undocumented_remedies() == ()


def test_the_worst_remedy_wins_over_a_set_of_codes():
    assert actions.device_remedy((63, 64)) == "replace"  # a failed remap outranks a recorded one
    assert actions.device_remedy((13, 31)) == "fix_application"
    assert actions.device_remedy((79,)) == "power_cycle"
    assert actions.device_remedy(()) == "none"


def test_an_unrecognized_code_is_never_repaired_by_guesswork():
    assert actions.xid_remedy(9999) == "investigate"
    assert actions.device_remedy((9999,)) == "investigate"
    # An unknown code alongside an application one must not read as "just fix the job".
    assert actions.device_remedy((13, 9999)) == "investigate"
    # ...but it must not downgrade a device that already needs replacing, either.
    assert actions.device_remedy((64, 9999)) == "replace"


def test_only_the_codes_that_corrupt_data_mark_results_untrusted():
    # The distinction that decides whether a job may retry past a fault. A device that fell
    # off the bus returned nothing; one that took a double-bit ECC error returned a wrong
    # number and kept going.
    assert actions.xid_untrusted(48) is True
    assert actions.xid_untrusted(95) is True
    assert actions.xid_untrusted(79) is False
    assert actions.xid_untrusted(9999) is False


def test_a_code_this_build_does_not_know_is_reported_rather_than_dropped(tmp_path):
    # The two failure modes here are symmetrical and both expensive. Guessing a severity for
    # an unseen code lets a driver release quarantine a fleet; dropping it silently lets the
    # same release become months of "those nodes are just flaky", with nothing anywhere naming
    # a code the vendor documents and this build does not.
    path = _log(
        tmp_path,
        [
            _record(1.0, "NVRM: Xid (PCI:0000:0c:00): 9999, something new"),
            _record(2.0, "NVRM: Xid (PCI:0000:1a:00): 79, fell off the bus"),
            _record(3.0, "NVRM: Xid (PCI:0000:1a:00): 13, graphics exception"),
        ],
    )
    events = xid.recent_xid_events(path)
    assert xid.xid_unclassified(events) == {"0000:0c:00.0": (9999,)}
    # And it stays out of both lists anything acts on.
    assert "0000:0c:00.0" not in xid.xid_fatal(events)
    assert "0000:0c:00.0" not in xid.xid_application_faults(events)


def test_codes_are_explained_in_the_words_an_operator_acts_on():
    assert "fallen off the bus" in actions.explain_codes((79,))
    assert actions.explain_codes(()) == ""


# --- node faults --------------------------------------------------------------------------


def test_an_oom_kill_is_recognized_in_both_spellings(tmp_path):
    path = _log(
        tmp_path,
        [
            _record(1.0, "Out of memory: Killed process 4242 (ray::map) total-vm:..."),
            _record(2.0, "oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=..."),
        ],
    )
    faults = kernel.node_faults(path, within_s=None)
    assert [f.kind for f in faults] == ["host_oom", "host_oom"]
    assert all(f.severity == "fatal" for f in faults)


def test_a_read_only_remount_is_fatal_because_every_later_spill_fails(tmp_path):
    path = _log(tmp_path, [_record(1.0, "EXT4-fs (nvme0n1p1): Remounting filesystem read-only")])
    (fault,) = kernel.node_faults(path, within_s=None)
    assert fault.kind == "filesystem_readonly"
    assert fault.severity == "fatal"


def test_a_corrected_pcie_error_is_a_note_and_an_uncorrected_one_is_not(tmp_path):
    path = _log(
        tmp_path,
        [
            _record(1.0, "pcieport 0000:00:01.0: AER: Corrected error received: 0000:0c:00.0"),
            _record(2.0, "pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received"),
        ],
    )
    faults = kernel.node_faults(path, within_s=None)
    assert [f.kind for f in faults] == ["pcie_fatal", "pcie_corrected"] or [
        f.kind for f in faults
    ] == ["pcie_corrected", "pcie_fatal"]
    assert kernel.worst_severity(faults) == "fatal"


def test_a_driver_that_never_brought_a_device_up_is_fatal(tmp_path):
    # Distinct from every Xid, which is what a *working* driver reports about a device. Here
    # the driver says it never got one, so the node comes up short and a collective sized for
    # the fleet waits forever for a rank that will never join.
    path = _log(tmp_path, [_record(1.0, "NVRM: RmInitAdapter failed! (0x26:0xffff:1223)")])
    (fault,) = kernel.node_faults(path, within_s=None)
    assert fault.kind == "driver_init"
    assert fault.severity == "fatal"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("nvme nvme0: I/O 322 QID 3 timeout, aborting", "storage_io"),
        ("nvme nvme0: resetting controller", "storage_io"),
        ("blk_update_request: I/O error, dev nvme0n1, sector 12345", "storage_io"),
        ("rcu: INFO: rcu_sched detected stalls on CPUs/tasks:", "rcu_stall"),
        ("cgroup: fork rejected by pids controller in /kubepods/burstable/podabc", "process_limit"),
    ],
)
def test_the_failures_that_precede_a_dead_node_are_recognized(tmp_path, message, kind):
    path = _log(tmp_path, [_record(1.0, message)])
    (fault,) = kernel.node_faults(path, within_s=None)
    assert fault.kind == kind


def test_every_category_has_a_severity():
    # A kind with no entry silently degrades to "note", which is the quiet way for a fatal
    # condition to stop reaching the drain list.
    kinds = {kind for kind, _ in kernel._PATTERNS}
    assert kinds <= set(kernel.SEVERITY_BY_KIND)


def test_ordinary_kernel_chatter_is_not_a_fault(tmp_path):
    path = _log(
        tmp_path,
        [
            _record(1.0, "systemd[1]: Started Session 3 of user ray."),
            _record(2.0, "audit: type=1400 audit(...): apparmor=DENIED"),
        ],
    )
    assert kernel.node_faults(path, within_s=None) == ()
    assert kernel.worst_severity(()) == "none"


def test_node_faults_age_out_of_their_window_too(tmp_path, frozen_clock):
    path = _log(
        tmp_path,
        [
            _record(_NOW_S - 86_400, "Out of memory: Killed process 1 (old)"),
            _record(_NOW_S, "watchdog: BUG: soft lockup - CPU#3 stuck for 22s!"),
        ],
    )
    assert [f.kind for f in kernel.node_faults(path, within_s=3600.0)] == ["lockup"]


def test_counts_are_what_a_rate_policy_reads(tmp_path):
    path = _log(
        tmp_path,
        [_record(float(i), f"pcieport: AER: Corrected error received: {i}") for i in range(7)],
    )
    assert kernel.node_fault_counts(kernel.node_faults(path, within_s=None)) == {
        "pcie_corrected": 7
    }


def test_an_unreadable_log_is_not_a_healthy_node(tmp_path):
    missing = str(tmp_path / "absent")
    assert kernel.node_faults(missing) == ()
    assert kernel.node_faults_readable(missing) is False
