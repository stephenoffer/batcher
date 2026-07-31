"""The fractional-scheduling vocabulary: bytes in, a schedulable device fraction out.

Pure arithmetic, so every case is constructed rather than probed. The properties that matter
are that rounding always goes *up* (a fraction is a reservation, and under-asking is the error
that costs a device OOM), that an unknown input reports no opinion rather than a fabricated
one, and that the three derived quantities — the co-tenancy, the device count, and the byte
share — stay consistent with the fraction they were derived from. The last is the one a
copy-paste would break silently: a device packed at `0.25` and admitted at four tenants must
grant each of them exactly a quarter of the usable bytes.
"""

from __future__ import annotations

import pytest

from batcher._internal.device_share import (
    MAX_COTENANTS,
    PACK_QUANTA,
    DeviceShare,
    balanced_fraction,
    cotenants_per_device,
    devices_for,
    fits_one_device,
    pack_fraction,
    quantize_fraction,
    share_bytes,
    usable_bytes,
)

pytestmark = pytest.mark.unit

GIB = 1 << 30


def test_the_ladder_is_ascending_and_ends_at_a_whole_device() -> None:
    assert list(PACK_QUANTA) == sorted(PACK_QUANTA)
    assert PACK_QUANTA[-1] == 1.0
    assert round(1.0 / PACK_QUANTA[0]) == MAX_COTENANTS


def test_usable_bytes_removes_the_headroom_and_never_goes_negative() -> None:
    assert usable_bytes(100, 0.15) == 85
    assert usable_bytes(100, 0.0) == 100
    assert usable_bytes(0, 0.15) == 0, "an unknown device has no usable bytes"
    assert usable_bytes(-5, 0.15) == 0


def test_a_headroom_beyond_nine_tenths_is_clamped_rather_than_believed() -> None:
    # A device nothing can run on is a config error, not a plan; the clamp keeps a
    # mistyped headroom from silently refusing every placement on the fleet.
    assert usable_bytes(100, 0.99) == 10
    assert usable_bytes(100, -1.0) == 100


def test_quantizing_always_rounds_up_to_the_next_schedulable_share() -> None:
    assert quantize_fraction(0.01) == 0.25
    assert quantize_fraction(0.25) == 0.25, "an exact quantum is not rounded past itself"
    assert quantize_fraction(0.26) == 0.5
    assert quantize_fraction(0.51) == 1.0
    assert quantize_fraction(1.0) == 1.0


def test_a_need_larger_than_one_device_becomes_whole_devices() -> None:
    assert quantize_fraction(1.01) == 2.0
    assert quantize_fraction(2.0) == 2.0
    assert quantize_fraction(6.1) == 7.0


def test_nothing_known_is_reported_as_no_opinion_not_as_a_guess() -> None:
    assert quantize_fraction(0.0) == 0.0
    assert quantize_fraction(-1.0) == 0.0
    assert pack_fraction(3 * GIB, 0) == 0.0, "an unreadable device packs nothing"
    assert pack_fraction(0, 80 * GIB) == 0.0, "an unmeasured need packs nothing"


def test_pack_fraction_sizes_a_small_claimant_onto_a_share_of_a_large_device() -> None:
    assert pack_fraction(3 * GIB, 80 * GIB) == 0.25
    assert pack_fraction(30 * GIB, 80 * GIB) == 0.5
    assert pack_fraction(60 * GIB, 80 * GIB) == 1.0


def test_pack_fraction_charges_the_headroom_before_dividing() -> None:
    # 0.5 of 80 GiB is 40, but only 68 GiB is usable at the default headroom, so a 40 GiB
    # claimant is 0.59 of what it may actually have and takes the whole device.
    assert pack_fraction(40 * GIB, 80 * GIB) == 1.0
    assert pack_fraction(40 * GIB, 80 * GIB, headroom=0.0) == 0.5


def test_share_bytes_is_the_inverse_of_the_fraction_that_produced_it() -> None:
    fraction = pack_fraction(3 * GIB, 80 * GIB)
    granted = share_bytes(80 * GIB, fraction)
    assert granted == int(usable_bytes(80 * GIB, 0.15) * fraction)
    assert granted >= 3 * GIB, "the granted share must cover the need it was chosen for"


def test_share_bytes_reports_nothing_for_an_undecided_fraction() -> None:
    assert share_bytes(80 * GIB, 0.0) == 0
    assert share_bytes(0, 0.25) == 0


def test_cotenants_follow_from_the_fraction() -> None:
    assert cotenants_per_device(0.25) == 4
    assert cotenants_per_device(0.5) == 2
    assert cotenants_per_device(1.0) == 1
    assert cotenants_per_device(2.0) == 1, "a multi-device claimant does not share"
    assert cotenants_per_device(0.0) == 0, "no decision means no count"


def test_device_count_packs_claimants_rather_than_counting_them() -> None:
    assert devices_for(0.25, 16) == 4, "sixteen quarter-device workers need four devices"
    assert devices_for(0.5, 3) == 2
    assert devices_for(1.0, 3) == 3
    assert devices_for(2.0, 3) == 6, "a two-device claimant multiplies"
    assert devices_for(0.25, 0) == 0
    assert devices_for(0.0, 8) == 0


def test_balanced_fraction_grants_the_largest_share_that_still_fits_them_all() -> None:
    assert balanced_fraction(1) == 1.0
    assert balanced_fraction(2) == 0.5
    assert balanced_fraction(3) == 0.25, "three at 0.5 would not fit one device"
    assert balanced_fraction(4) == 0.25


def test_asking_for_more_cotenants_than_the_ladder_divides_hits_the_floor() -> None:
    # Refused rather than approximated: the extra tenants queue, and the caller should see
    # the ceiling it hit instead of a fraction that pretends they all ran.
    assert balanced_fraction(MAX_COTENANTS + 4) == min(PACK_QUANTA)


def test_balanced_and_cotenant_counts_are_mutually_consistent() -> None:
    for n in range(1, MAX_COTENANTS + 1):
        assert cotenants_per_device(balanced_fraction(n)) >= n


def test_fits_one_device_routes_an_unknown_device_to_the_sharded_path() -> None:
    assert fits_one_device(3 * GIB, 80 * GIB)
    assert not fits_one_device(80 * GIB, 80 * GIB), "headroom is not available to a claimant"
    assert not fits_one_device(3 * GIB, 0), "unknown routes to sharding, which is always correct"
    assert not fits_one_device(0, 80 * GIB)


def test_device_share_reports_whether_it_decided_anything() -> None:
    assert DeviceShare(0.25).is_fractional
    assert DeviceShare(0.25).decided
    assert not DeviceShare(1.0).is_fractional
    assert not DeviceShare(0.0).decided
    assert DeviceShare(0.5).devices == 1
