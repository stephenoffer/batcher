"""Unit tests for the distributed Ray-init kwargs + self-ship runtime_env.

These cover the control-plane decision of *how* batcher attaches to Ray and ships
its data plane to workers — pure kwargs construction, no Ray process required. The
flight-worker actors import `batcher` + its native extension to run; on a cluster
whose image doesn't already carry batcher (a source/editable install) that import
fails unless the package is uploaded via `runtime_env={"py_modules": [...]}`.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import active_config, set_config
from batcher.dist.executors.ray_runtime import lifecycle

pytestmark = pytest.mark.unit


@pytest.fixture
def restore_config():
    saved = active_config()
    yield
    set_config(saved)


def _with_distributed(**overrides):
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides)))


# Managed-cluster / address signals that route `_ray_init_kwargs` to attach vs local.
_CLUSTER_SIGNAL_VARS = ("RAY_ADDRESS", "ANYSCALE_SESSION_ID", "ANYSCALE_CLUSTER_ID")


@pytest.fixture
def _no_cluster_signal(monkeypatch):
    """A plain host with no address/managed-cluster signal — the genuine local path."""
    for var in _CLUSTER_SIGNAL_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_self_ship_uploads_a_source_install(monkeypatch):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    env = lifecycle._self_ship_runtime_env()
    # Ships the driver's package and pins pip off — workers use their base env for other
    # deps, and a managed hook can't inject an unresolvable per-job pip (e.g. batcher-engine).
    assert env == {"py_modules": ["/repo/python/batcher"], "pip": None}


def test_self_ship_uploads_a_site_packages_install_too(monkeypatch):
    # A pip/site-packages install is shipped too: a driver that pip-installed batcher and
    # attached to an arbitrary cluster cannot assume that cluster's image carries a compatible
    # batcher, so shipping guarantees driver==worker code (the old skip-for-site-packages
    # heuristic produced a silent ModuleNotFoundError on the local-install → remote-cluster case).
    monkeypatch.setattr(
        "batcher.__file__", "/opt/conda/lib/python3.12/site-packages/batcher/__init__.py"
    )
    assert lifecycle._self_ship_runtime_env() == {
        "py_modules": ["/opt/conda/lib/python3.12/site-packages/batcher"],
        "pip": None,
    }


def test_self_ship_skipped_only_when_cluster_image_is_trusted(monkeypatch, restore_config):
    # The one opt-out: a production image that bakes a matching batcher into every node. No
    # upload, but pip is still pinned off so a managed hook can't inject a broken per-job pip.
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _with_distributed(trust_cluster_image=True)
    assert lifecycle._self_ship_runtime_env() == {"pip": None}


def test_init_kwargs_attach_auto_ships_when_runtime_env_unset(monkeypatch, restore_config):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _with_distributed(ray_address="auto", runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=4)
    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs  # never pin CPUs against a real cluster
    assert kwargs["runtime_env"] == {"py_modules": ["/repo/python/batcher"], "pip": None}


def test_explicit_runtime_env_wins_over_auto_ship(monkeypatch, restore_config):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    explicit = {"pip": ["numpy"]}
    _with_distributed(ray_address="auto", runtime_env=explicit)
    assert lifecycle._ray_init_kwargs(workers=4)["runtime_env"] == explicit


def test_local_cluster_does_not_auto_ship(_no_cluster_signal, restore_config):
    _no_cluster_signal.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _with_distributed(ray_address=None, runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=3)
    # Spinning a local in-process cluster: cap CPUs, ship nothing (workers are local).
    assert kwargs["num_cpus"] == 3
    assert "runtime_env" not in kwargs


def test_managed_cluster_attaches_without_ray_address(_no_cluster_signal, restore_config):
    # A managed workspace (Anyscale) that exports no RAY_ADDRESS must still ATTACH to the
    # running cluster — a bare local start would strand a distributed job on one node.
    _no_cluster_signal.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _no_cluster_signal.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _with_distributed(ray_address=None, runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=3)
    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs  # never pin CPUs against a real cluster
    assert kwargs["runtime_env"] == {"py_modules": ["/repo/python/batcher"], "pip": None}


def test_force_local_overrides_managed_detection(_no_cluster_signal, restore_config):
    # The reachability fallback: even on a managed workspace, `force_local` starts a local
    # single-node Ray (used when `address="auto"` found no reachable cluster).
    _no_cluster_signal.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _no_cluster_signal.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _with_distributed(ray_address="auto", runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=5, force_local=True)
    assert "address" not in kwargs
    assert kwargs["num_cpus"] == 5
