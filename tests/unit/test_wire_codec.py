"""Choosing the shuffle's wire codec against the link it will cross.

The failure this removes is silent and expensive: on a fast fabric a compressor that cannot
keep up becomes the ceiling, so the node someone paid extra for runs its shuffle at compressor
speed with every counter reading healthy.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.transfer.codec import CODEC_CODES, codec_for_fabric, resolve_codec

pytestmark = pytest.mark.unit


def test_a_very_slow_link_wants_the_highest_ratio() -> None:
    """A 1 Gb/s wire moves 125 MB/s: the cores are idle and every removed byte is a saving."""
    assert codec_for_fabric(1.0, cores=16.0) == "zstd"


def test_a_mid_range_nic_wants_the_fast_codec() -> None:
    assert codec_for_fabric(10.0, cores=16.0) == "lz4"
    assert codec_for_fabric(25.0, cores=32.0) == "lz4"


def test_a_fast_fabric_wants_no_compression_at_all() -> None:
    """No codec keeps up with 400 Gb/s, so every one of them is a ceiling below the wire."""
    assert codec_for_fabric(400.0, cores=64.0) == "none"


def test_more_cores_move_the_crossover_up() -> None:
    """The trade is cores against wire, so a worker with cores to spare compresses for longer."""
    assert codec_for_fabric(100.0, cores=32.0) == "none"
    assert codec_for_fabric(100.0, cores=256.0) == "lz4"


def test_the_compressor_is_not_assumed_to_own_the_machine() -> None:
    """Handing it every core concludes that compression wins on every fabric, which it does not."""
    assert codec_for_fabric(400.0, cores=64.0) == "none"


def test_one_core_cannot_keep_up_with_a_modern_fabric() -> None:
    """The conservative default: an uninformed caller does not compress a fast wire."""
    assert codec_for_fabric(25.0, cores=1.0) == "none"


def test_an_unknown_fabric_keeps_the_shipped_default() -> None:
    assert codec_for_fabric(0.0) == "lz4"
    assert codec_for_fabric(25.0, cores=0.0) == "lz4"


def test_an_explicit_codec_is_never_overruled_by_a_measurement() -> None:
    """Somebody made that decision about their own deployment."""
    assert resolve_codec("zstd", fabric_gbps=400.0) == CODEC_CODES["zstd"]
    assert resolve_codec("none", fabric_gbps=1.0) == CODEC_CODES["none"]
    assert resolve_codec("lz4", fabric_gbps=400.0) == CODEC_CODES["lz4"]


def test_auto_decides_against_the_measured_rate() -> None:
    assert resolve_codec("auto", fabric_gbps=400.0, cores=64.0) == CODEC_CODES["none"]
    assert resolve_codec("auto", fabric_gbps=10.0, cores=16.0) == CODEC_CODES["lz4"]
    assert resolve_codec("auto", fabric_gbps=1.0, cores=16.0) == CODEC_CODES["zstd"]


def test_auto_on_an_unreadable_fabric_is_the_shipped_default() -> None:
    assert resolve_codec("auto") == CODEC_CODES["lz4"]


def test_an_unrecognized_name_resolves_as_the_call_site_always_did() -> None:
    assert resolve_codec("brotli", fabric_gbps=10.0) == CODEC_CODES["lz4"]
    assert resolve_codec("") == CODEC_CODES["lz4"]


def test_the_name_is_case_and_space_insensitive() -> None:
    assert resolve_codec(" AUTO ", fabric_gbps=400.0, cores=64.0) == CODEC_CODES["none"]


def test_the_codes_match_the_engine_the_call_site_talks_to() -> None:
    """The mapping was inline at the call site; a second spelling is a silent misselection."""
    assert CODEC_CODES == {"none": 0, "lz4": 1, "zstd": 2}
