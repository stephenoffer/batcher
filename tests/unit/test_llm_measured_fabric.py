"""Tensor-parallel advice that reads the links, not only the model name.

An SXM board whose NVLink has dropped is a PCIe card that every nameplate check calls an
NVLink one. The throughput loss is the same 30-50% the PCIe warning already exists for, with
none of the visibility, and the fix is different: drain the node rather than change the
degree. These tests pin that third case, and pin that a node whose cards have no NVLink at all
is reported as unknown rather than as down.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PerformanceWarning
from batcher._internal.hardware.fabric.nvlink import NvLinkStatus
from batcher.ml.llm.engines import parallelism

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _unwarn(monkeypatch):
    """The warning fires once per process; reset it so each test sees its own."""
    monkeypatch.setattr(parallelism, "_TP_WARNED", False)


def _links(monkeypatch, statuses):
    monkeypatch.setattr("batcher._internal.hardware.fabric.nvlink_status", lambda: tuple(statuses))


def test_a_node_with_no_nvlink_reports_unknown_not_down(monkeypatch):
    # "Down" would send an operator to inspect a fabric the hardware never had.
    _links(monkeypatch, [NvLinkStatus(index=0, links=0), NvLinkStatus(index=1, links=0)])
    assert parallelism.measured_link_class() == "unknown"
    _links(monkeypatch, [])
    assert parallelism.measured_link_class() == "unknown"


def test_every_link_up_is_nvlink(monkeypatch):
    _links(monkeypatch, [NvLinkStatus(index=i, links=18, active_links=18) for i in range(8)])
    assert parallelism.measured_link_class() == "nvlink"


def test_links_down_read_as_pcie(monkeypatch):
    _links(monkeypatch, [NvLinkStatus(index=i, links=18, active_links=0) for i in range(8)])
    assert parallelism.measured_link_class() == "pcie"


def test_one_device_off_the_fabric_costs_the_group_the_fabric(monkeypatch):
    # A collective runs at the rate of the slowest pair in its group, so a partially-down
    # fabric is not a third state between nvlink and pcie.
    _links(
        monkeypatch,
        [
            NvLinkStatus(index=0, links=18, active_links=18),
            NvLinkStatus(index=1, links=18, active_links=4),
        ],
    )
    assert parallelism.measured_link_class() == "pcie"


def test_a_down_fabric_on_an_nvlink_card_is_called_out_as_a_node_fault(monkeypatch):
    _links(monkeypatch, [NvLinkStatus(index=i, links=18, active_links=0) for i in range(2)])
    with pytest.warns(PerformanceWarning, match="NVLink fabric is reported DOWN"):
        parallelism.warn_about_tensor_parallelism(2, 10.0, 80.0, "NVIDIA A100-SXM4-80GB")


def test_a_healthy_fabric_on_an_nvlink_card_says_nothing(monkeypatch, recwarn):
    _links(monkeypatch, [NvLinkStatus(index=i, links=18, active_links=18) for i in range(2)])
    parallelism.warn_about_tensor_parallelism(2, 10.0, 80.0, "NVIDIA A100-SXM4-80GB")
    assert [w for w in recwarn if issubclass(w.category, PerformanceWarning)] == []


def test_an_unreadable_fabric_says_nothing(monkeypatch, recwarn):
    # Absent telemetry must not produce a node-fault warning about hardware nobody probed.
    _links(monkeypatch, [])
    parallelism.warn_about_tensor_parallelism(2, 10.0, 80.0, "NVIDIA A100-SXM4-80GB")
    assert [w for w in recwarn if issubclass(w.category, PerformanceWarning)] == []


def test_the_model_that_cannot_fit_still_wins_the_warning(monkeypatch):
    # Ordering matters: "it will not load" is more urgent than "it will be slow".
    _links(monkeypatch, [NvLinkStatus(index=i, links=18, active_links=0) for i in range(2)])
    with pytest.warns(PerformanceWarning, match="smallest group that fits"):
        parallelism.warn_about_tensor_parallelism(2, 400.0, 80.0, "NVIDIA A100-SXM4-80GB")
