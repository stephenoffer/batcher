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


def test_self_ship_uploads_a_source_install(monkeypatch):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    env = lifecycle._self_ship_runtime_env()
    assert env == {"py_modules": ["/repo/python/batcher"]}


def test_self_ship_skips_a_site_packages_install(monkeypatch):
    monkeypatch.setattr(
        "batcher.__file__", "/opt/conda/lib/python3.12/site-packages/batcher/__init__.py"
    )
    assert lifecycle._self_ship_runtime_env() is None


def test_init_kwargs_attach_auto_ships_when_runtime_env_unset(monkeypatch, restore_config):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    _with_distributed(ray_address="auto", runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=4)
    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs  # never pin CPUs against a real cluster
    assert kwargs["runtime_env"] == {"py_modules": ["/repo/python/batcher"]}


def test_explicit_runtime_env_wins_over_auto_ship(monkeypatch, restore_config):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    explicit = {"pip": ["numpy"]}
    _with_distributed(ray_address="auto", runtime_env=explicit)
    assert lifecycle._ray_init_kwargs(workers=4)["runtime_env"] == explicit


def test_local_cluster_does_not_auto_ship(monkeypatch, restore_config):
    monkeypatch.setattr("batcher.__file__", "/repo/python/batcher/__init__.py")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    _with_distributed(ray_address=None, runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=3)
    # Spinning a local in-process cluster: cap CPUs, ship nothing (workers are local).
    assert kwargs["num_cpus"] == 3
    assert "runtime_env" not in kwargs
