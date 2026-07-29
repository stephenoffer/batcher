"""Three Carbonite policies that answered confidently when they had no answer.

Each of these is pure arithmetic over inputs a caller supplies, and each had a case where
the honest output is "I cannot tell" or "there is nothing to do" — and instead returned a
plausible number that reads as a decision. That is the shape of failure worth testing for
here: not a crash, but a wrong answer that looks like a right one and is acted on far from
where it was produced.

- `select_mode` read two *unknown* addresses as equal and called that same-process.
- `locality_ratio_counts` could exceed 1.0 from a double-counted counter, and the tuning
  loop reads it as a fraction.
- `assign_reducer_hosts` named actor `0` of a fleet with no actors.

`partitions_for_envelope` is the fourth of the same family, but its answer cannot be fixed
by clamping — the shortfall is inherent — so `envelope_shortfall` reports it instead.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies.spill_shape import (
    MAX_SPILL_PARTITIONS,
    envelope_shortfall,
    partitions_for_envelope,
)
from batcher.carbonite.transfer.locality import (
    TransferMode,
    locality_ratio,
    locality_ratio_counts,
    select_mode,
)
from batcher.carbonite.transfer.placement import assign_reducer_hosts, reducer_affinity

pytestmark = pytest.mark.unit


# --- select_mode: an unknown address is not a match ---------------------------


def test_two_unknown_addresses_are_not_the_same_process() -> None:
    """The bug: `"" == ""` is both being absent, not both being here.

    Read as `DIRECT_MEMORY` the fetcher looks in a local store that does not hold the
    bucket, and the symptom is a missing partition rather than the address bug it is.
    """
    assert select_mode("", "") is TransferMode.NETWORK


def test_one_unknown_address_is_not_a_match_either() -> None:
    assert select_mode("", "10.0.0.1:5005") is TransferMode.NETWORK
    assert select_mode("10.0.0.1:5005", "") is TransferMode.NETWORK


def test_two_unknown_nodes_are_not_the_same_host() -> None:
    """Same reasoning one level down: empty node ids must not imply co-location."""
    assert select_mode("a:1", "b:2", source_node="", local_node="") is TransferMode.NETWORK


def test_a_real_match_still_resolves_to_the_fast_path() -> None:
    """The fix must not cost the case the whole module exists for."""
    assert select_mode("10.0.0.1:5005", "10.0.0.1:5005") is TransferMode.DIRECT_MEMORY
    assert (
        select_mode("a:1", "b:2", source_node="node-7", local_node="node-7")
        is TransferMode.SHARED_MEMORY
    )


def test_unknown_resolves_to_the_mode_that_is_always_correct() -> None:
    """`NETWORK` is never wrong, only slower — which is the right default for a guess."""
    assert not select_mode("", "").is_local


# --- locality_ratio_counts: a fraction that stays a fraction ------------------


def test_a_double_counted_counter_cannot_report_above_one() -> None:
    """The tuning loop reads this as a fraction; 1.3 looks like a strong signal, not noise."""
    assert locality_ratio_counts(130, 100) == 1.0


def test_a_negative_counter_cannot_report_below_zero() -> None:
    assert locality_ratio_counts(-5, 100) == 0.0


def test_the_ordinary_case_is_unchanged() -> None:
    assert locality_ratio_counts(30, 100) == pytest.approx(0.3)
    assert locality_ratio_counts(0, 100) == 0.0
    assert locality_ratio_counts(100, 100) == 1.0


def test_no_fetches_reads_as_fully_local() -> None:
    """Matching `locality_ratio`'s convention: nothing crossed the network, because nothing went."""
    assert locality_ratio_counts(0, 0) == 1.0
    assert locality_ratio([]) == 1.0


# --- assign_reducer_hosts: nowhere to put anything ----------------------------


def test_no_actors_places_nothing() -> None:
    """`[0, 0, 0]` names actor 0 of a fleet that has none."""
    assert assign_reducer_hosts(3, [], {}) == []


def test_no_reducers_places_nothing() -> None:
    assert assign_reducer_hosts(0, ["n1", "n2"], {}) == []


def test_the_default_round_robin_is_unchanged() -> None:
    """An unskewed shuffle must place exactly as it did before."""
    assert assign_reducer_hosts(5, ["n1", "n2"], {}) == [0, 1, 0, 1, 0]


def test_an_affine_bucket_lands_on_its_node() -> None:
    hosts = assign_reducer_hosts(4, ["n1", "n2", "n2"], {0: "n2", 1: "n2"})
    assert hosts[0] in (1, 2)
    assert hosts[1] in (1, 2)
    assert hosts[0] != hosts[1], "both reducers piled onto one actor of the node"


def test_an_affinity_naming_an_unknown_node_falls_back() -> None:
    """A node that holds no actors must not strand the reducer."""
    hosts = assign_reducer_hosts(2, ["n1", "n2"], {0: "n-gone"})
    assert hosts == [0, 1]


def test_a_uniform_bucket_is_not_called_concentrated() -> None:
    """50/50 across two nodes is the case a naive `>= 0.5` threshold gets wrong."""
    assert reducer_affinity({0: {"n1": 50, "n2": 50}}) == {}


def test_a_genuinely_skewed_bucket_is_flagged() -> None:
    assert reducer_affinity({0: {"n1": 90, "n2": 10}}) == {0: "n1"}


# --- partitions_for_envelope: the shortfall it cannot clamp away --------------


def test_a_normal_state_shards_to_fit() -> None:
    """Below the cap the function delivers what its name promises."""
    assert envelope_shortfall(1 << 30, 1 << 20) == 0
    assert partitions_for_envelope(1 << 30, 1 << 20) == 1024


def test_past_the_cap_each_bucket_exceeds_the_envelope() -> None:
    """The promise breaks and nothing said so; now `envelope_shortfall` does.

    At 1 PiB of state against a 1 GiB envelope the count saturates at 4,096, so each bucket
    is 256 GiB — 255 GiB over target. Safe only because the reduce re-partitions an
    over-large bucket by grace recursion, which is a re-read, not a failure.
    """
    assert partitions_for_envelope(1 << 50, 1 << 30) == MAX_SPILL_PARTITIONS
    assert envelope_shortfall(1 << 50, 1 << 30) == 255 * (1 << 30)


def test_the_shortfall_is_zero_for_unusable_input() -> None:
    """An un-sized plan has no shortfall to report, only no answer."""
    assert envelope_shortfall(0, 1 << 20) == 0
    assert envelope_shortfall(1 << 30, 0) == 0
