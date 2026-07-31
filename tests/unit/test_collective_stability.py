"""Making a multi-GPU collective fail rather than hang.

A collective's default failure mode is the one a scheduler cannot survive: when a rank dies or
a device faults, the surviving ranks do not raise — they sit in the collective holding their
GPUs. From the orchestrator's side nothing has failed. The task is running, the actor is alive,
and no progress is being made, so every recovery mechanism in the package, all of which are
downstream of a failure being *reported*, never runs at all.

The tests below fix the two things that keeps it honest: the settings that turn that hang into
an ordinary task failure, and the rule that an operator's own choice is never overwritten.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import collectives as co

pytestmark = pytest.mark.unit


def test_an_empty_environment_gets_the_settings_that_prevent_a_hang():
    env = co.stability_env({})
    # Both spellings, because PyTorch renamed the variable and a cluster mid-upgrade runs
    # workers on either side of the rename — which is exactly when a hang is most likely.
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["NCCL_DEBUG"] == "WARN"


def test_the_operators_own_setting_is_never_replaced():
    env = co.stability_env({"NCCL_DEBUG": "INFO"})
    assert "NCCL_DEBUG" not in env
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" in env


def test_a_fully_configured_fleet_gets_nothing_added():
    settled = dict.fromkeys(co.STABILITY_VARS, "1")
    assert co.stability_env(settled) == {}


def test_the_headline_finding_is_that_a_fault_would_hang():
    findings = co.collective_findings({}, rdma_ports=0)
    assert findings
    assert "hang" in findings[0]


def test_async_error_handling_in_either_spelling_settles_the_finding():
    for key in ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_ASYNC_ERROR_HANDLING"):
        findings = co.collective_findings({key: "1", "NCCL_DEBUG": "WARN"}, rdma_ports=0)
        assert findings == ()


def test_disabling_infiniband_is_only_a_finding_where_there_is_a_fabric_to_lose():
    env = {"NCCL_IB_DISABLE": "1", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "NCCL_DEBUG": "WARN"}
    assert co.collective_findings(env, rdma_ports=0) == ()
    # On a node whose `/sys` is not mounted the port count reads zero, so this must be
    # conditional on *knowing* there is a fabric — telling an operator to re-enable a
    # transport their node may not have is worse than saying nothing.
    assert any("RDMA" in f for f in co.collective_findings(env, rdma_ports=4))


def test_shared_memory_disabled_is_reported():
    env = {"NCCL_SHM_DISABLE": "true", "NCCL_ASYNC_ERROR_HANDLING": "1", "NCCL_DEBUG": "WARN"}
    assert any("shared" in f for f in co.collective_findings(env, rdma_ports=0))


def test_a_flag_set_to_zero_is_not_treated_as_enabled():
    env = {"NCCL_SHM_DISABLE": "0", "NCCL_ASYNC_ERROR_HANDLING": "1", "NCCL_DEBUG": "WARN"}
    assert co.collective_findings(env, rdma_ports=0) == ()


def test_the_stability_and_topology_variable_sets_never_contend():
    # Two modules write collective environment variables: this one decides what happens when
    # the wires break, the other decides which wires to use. Sharing a name would let one
    # silently overwrite the other's decision depending on merge order.
    from batcher.dist.gpu.fabric.collective_env import COLLECTIVE_VARS

    assert not set(co.STABILITY_VARS) & set(COLLECTIVE_VARS)
