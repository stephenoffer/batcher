"""Machine-scoped learned state names the machine that will run the work, not the driver.

`metadata.hardware_scope.scoped` keys anything measured in machine units by a hardware
fingerprint, so unlike machines do not blend. It always used *this process's* class, which
splits cleanly in two:

* Written and read on the process that executed the work — the UDF, autobatch and device
  loops. The local class is the right one, and nothing here changes them.
* Written and read on the **driver**, about work done on the **workers** — the join-strategy
  bandit, the broadcast and sort-merge crossovers, the build-side priors. Those name the wrong
  machine. A fleet that autoscales from one worker type to another files both under one key,
  and two drivers of different classes against identical workers fragment one model into two.

The fix is a scope rather than a parameter, and that choice is the thing most worth pinning.
The failure mode of getting this wrong is not a blend, it is **silence**: if the write and the
read resolve the class independently — say by asking the cluster twice, either side of an
autoscale — every value is filed under a key nothing will ever read, and the loop stops
accruing with no error anywhere. One ambient scope over both halves cannot disagree with
itself.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware import fingerprint
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hardware_scope import planning_for, scoped

pytestmark = pytest.mark.unit

_WORKERS = "ffffffffffff"
_OTHER = "aaaaaaaaaaaa"


def test_the_default_is_this_machine():
    """Every existing caller passes nothing and must be unmoved."""
    assert scoped("ns").endswith(f"@{fingerprint()}")


def test_a_scope_renames_the_namespace():
    with planning_for(_WORKERS):
        assert scoped("ns") == f"ns@{_WORKERS}"


def test_the_scope_is_restored_on_exit():
    """A leaked scope would silently re-key every later query in the process."""
    before = scoped("ns")
    with planning_for(_WORKERS):
        pass
    assert scoped("ns") == before


def test_the_scope_is_restored_even_when_the_body_raises():
    before = scoped("ns")
    with pytest.raises(RuntimeError), planning_for(_WORKERS):
        raise RuntimeError("boom")
    assert scoped("ns") == before


def test_an_empty_fingerprint_is_a_no_op():
    """A single-node run and an unprobeable fleet both pass `""`."""
    before = scoped("ns")
    with planning_for(""):
        assert scoped("ns") == before


def test_an_explicit_argument_outranks_the_scope():
    """The scope is the default, not an override — a caller that knows better still wins."""
    with planning_for(_WORKERS):
        assert scoped("ns", _OTHER) == f"ns@{_OTHER}"


def _arm_evidence(hub, signature: str) -> float:
    """Total observations the bandit holds for `signature` under the *current* scope."""
    from batcher.kyber.learned_tuning.bandit import _NS_ARM

    stats = hub.get_keyed_param(scoped(_NS_ARM), signature) or {}
    return sum(float(s.get("n", 0.0)) for s in stats.values() if isinstance(s, dict))


def test_a_write_and_a_read_inside_one_scope_agree():
    """The property the whole design exists for, exercised through the real bandit.

    Asserted on *evidence*, not on which arm wins: UCB1 deliberately gives an untried arm a
    turn, so the argmax is a statement about exploration rather than about the key. What the
    key controls is whether the observations are visible at all.
    """
    from batcher.kyber.learned_tuning.bandit import record_join_strategy

    hub = MetadataHub(InProcessBackend())
    with planning_for(_WORKERS):
        for _ in range(40):
            record_join_strategy(hub, "sig", "broadcast", wall_ms=5.0, input_rows=1e6)
        assert _arm_evidence(hub, "sig") > 0.0
    # Outside the scope the driver's own class has learned nothing about this signature,
    # which is what proves the key moved rather than the scope being decorative.
    assert _arm_evidence(hub, "sig") == 0.0


def test_two_worker_classes_do_not_blend():
    """The autoscaling case: one driver, two worker generations, one metadata store."""
    from batcher.kyber.learned_tuning.bandit import _NS_ARM, record_join_strategy

    hub = MetadataHub(InProcessBackend())
    with planning_for(_WORKERS):
        for _ in range(40):
            record_join_strategy(hub, "sig", "broadcast", wall_ms=5.0, input_rows=1e6)
    with planning_for(_OTHER):
        for _ in range(40):
            record_join_strategy(hub, "sig", "sort_merge", wall_ms=5.0, input_rows=1e6)
    with planning_for(_WORKERS):
        arms = hub.get_keyed_param(scoped(_NS_ARM), "sig") or {}
        assert set(arms) == {"broadcast"}, "the other class's arm must not be visible here"
    with planning_for(_OTHER):
        arms = hub.get_keyed_param(scoped(_NS_ARM), "sig") or {}
        assert set(arms) == {"sort_merge"}


def test_the_crossover_loop_is_scoped_too():
    """The bandit is not the only driver-side learner; the crossovers key the same way."""
    from batcher.kyber.learned_tuning.crossover import _NS_BCAST, record_broadcast_timing

    hub = MetadataHub(InProcessBackend())
    with planning_for(_WORKERS):
        for i in range(60):
            record_broadcast_timing(hub, "broadcast", float(i) * 1e5, 10.0 + i)
        assert hub.get_keyed_param(scoped(_NS_BCAST), "broadcast")
    assert not hub.get_keyed_param(scoped(_NS_BCAST), "broadcast")


def test_a_single_node_run_keys_everything_locally():
    """End to end through the conductor: no cluster, so nothing is re-keyed."""
    import batcher as bt

    before = scoped("kyber.probe")
    bt.from_pydict({"a": [1, 2, 3]}).filter(bt.col("a") > 1).collect()
    assert scoped("kyber.probe") == before
