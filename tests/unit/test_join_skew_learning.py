"""Metadata-driven join-skew learning: hot keys measured by the detection pre-pass
are persisted by join shape and reused on later runs, so salting engages without
re-running the pre-pass. The loop is result-preserving (salting changes scheduling,
not the joined relation), so these tests pin the persistence semantics; the
distributed equivalence suite proves correctness end to end.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.skew import (
    join_skew_key,
    load_learned_hot_keys,
    load_learned_skew,
    persist_hot_keys,
    resolve_hot_keys,
    salt_factor,
)

pytestmark = pytest.mark.unit


def _join_plan():
    a = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = pa.table({"id": [1, 2, 3], "w": [1, 2, 3]})
    return bt.from_arrow(a).join(bt.from_arrow(b), on="id")._plan


def test_join_skew_key_is_stable_and_shape_specific():
    plan = _join_plan()
    k1 = join_skew_key("LIR", "RIR", plan)
    k2 = join_skew_key("LIR", "RIR", plan)
    assert k1 == k2 and len(k1) == 16  # deterministic short hash
    # A different shape (different side IR) keys differently.
    assert join_skew_key("OTHER", "RIR", plan) != k1


def test_learned_hot_keys_round_trip_and_none_vs_empty():
    plan = _join_plan()
    key = join_skew_key("LIR", "RIR", plan)

    # Never measured → None (so the caller knows to run the pre-pass).
    assert load_learned_hot_keys(key) is None

    # A measured non-empty hot set round-trips and is what later runs salt on.
    persist_hot_keys(key, ["7", "42"])
    assert load_learned_hot_keys(key) == ["7", "42"]

    # A measured EMPTY result ("not skewed") is distinct from never-measured, so a
    # non-skewed shape never re-runs the pre-pass.
    empty_key = join_skew_key("LIR2", "RIR2", plan)
    persist_hot_keys(empty_key, [])
    assert load_learned_hot_keys(empty_key) == []
    assert load_learned_hot_keys(empty_key) is not None


def test_the_measured_share_is_persisted_and_sizes_the_fan_out():
    """The fan-out is sized from the hot key's **measured** share, not from the threshold
    at which a value starts counting as hot.

    `salt_factor` implements `s >= f x P`, which is only the right answer when `f` is what
    the key actually holds. Fed the detection *threshold* instead (0.10 by default) it
    returns `ceil(0.10 x 8) = 1` for every key however skewed, floored to 2 — so a value
    holding 40% of a side across 8 reducers was fanned 2 ways where it needs 4, and stayed
    2.5x over the average reducer. Measured on an 8-worker cluster, that was the difference
    between the default path running ~12.5 s and an explicitly-salted one running ~1.9 s.
    """
    plan = _join_plan()
    key = join_skew_key("SHARE", "SHARE", plan)
    persist_hot_keys(key, ["0"], 0.40)
    assert load_learned_skew(key) == (["0"], 0.40)

    # The threshold answers the wrong question; the measurement answers the right one.
    assert salt_factor(0.10, 8) == 2
    assert salt_factor(0.40, 8) == 4

    hot, salt = resolve_hot_keys(
        plan, [], key, fraction=0.10, partitions=8, salt=0, detect=_unreachable
    )
    assert hot == ["0"]
    assert salt == 4, "the fan-out must come from the 40% measured, not the 10% threshold"


def test_a_legacy_record_without_a_share_falls_back_to_the_threshold():
    """A record written before the share was stored reads back as share-unknown, and the
    threshold then stands as the conservative floor it always was — an old record must not
    become a wrong number."""
    from batcher.core import default_hub

    plan = _join_plan()
    key = join_skew_key("LEGACY", "LEGACY", plan)
    default_hub().put_keyed_param("dist.skew", key, ["9"])  # the pre-share list form
    assert load_learned_skew(key) == (["9"], 0.0)
    hot, salt = resolve_hot_keys(
        plan, [], key, fraction=0.25, partitions=8, salt=0, detect=_unreachable
    )
    assert (hot, salt) == (["9"], salt_factor(0.25, 8))


def test_a_negative_salt_is_the_off_switch():
    """`skew_join_salt` is a fan-out, not a flag, and `0` has not meant "off" since a
    *measured* hot key started engaging salting on its own. A user who wants the plain
    co-partition shuffle pinned therefore needs a spelling that is not `0`, and it has to
    cost nothing — not even the hub lookup — or "never salt" is just a different bill."""
    plan = _join_plan()
    key = join_skew_key("OFF", "OFF", plan)
    persist_hot_keys(key, ["0"], 0.40)  # a known-skewed shape, which 0 would salt

    assert resolve_hot_keys(
        plan, [], key, fraction=0.10, partitions=8, salt=0, detect=_unreachable
    ) == (["0"], 4)
    assert resolve_hot_keys(
        plan, [], key, fraction=0.10, partitions=8, salt=-1, detect=_unreachable
    ) == ([], 0)


def test_a_negative_salt_is_accepted_by_config_validation():
    from batcher.config import Config, DistributedConfig

    assert Config().replace(distributed=DistributedConfig(skew_join_salt=-1)).validate()
    with pytest.raises(Exception, match="skew_join_salt"):
        Config().replace(distributed=DistributedConfig(skew_join_salt=-2)).validate()


def _unreachable():
    raise AssertionError("the pre-pass must not run when the shape is already learned")
