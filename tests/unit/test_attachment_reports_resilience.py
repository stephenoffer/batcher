"""The one-time Ray attachment line states what losing a node will cost.

The resilience profile is auto-selected from environment signals, so a preemptible site
Batcher does not recognize silently keeps the stable-cluster budgets, and the default
`shuffle_replication = 1` gives no bucket a copy. Both are facts a user only discovers after
a loss, so they are reported where the cluster is first seen.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from batcher.config import Config, config_context

pytestmark = pytest.mark.unit


def test_the_attachment_line_carries_resilience_and_replication(monkeypatch, caplog):
    from batcher.dist.executors.ray_runtime import lifecycle

    monkeypatch.setattr(lifecycle, "_reported_session", "")
    monkeypatch.setattr(lifecycle, "ray_session_key", lambda: "s1")
    monkeypatch.setattr(
        lifecycle, "cluster_topology", lambda: {"nodes": 3, "cpus": 24.0, "gpus": 0.0}
    )
    monkeypatch.setattr(lifecycle, "job_ships_batcher", lambda: True)

    class _Ray:
        @staticmethod
        def get_runtime_context():
            return object()

    base = Config()
    cfg = base.replace(
        distributed=dataclasses.replace(base.distributed, resilience="spot", shuffle_replication=1)
    )
    with config_context(cfg), caplog.at_level(logging.INFO, logger="batcher.dist"):
        lifecycle._report_attachment(_Ray())
    from batcher._internal.logging import _FIELDS_ATTR

    said = [r for r in caplog.records if r.getMessage() == "attached to Ray"]
    assert len(said) == 1
    fields = getattr(said[0], _FIELDS_ATTR)
    assert fields["resilience"] == "spot"
    assert fields["shuffle_replication"] == 1
