"""Unit tests for the distributed Ray-init kwargs + self-ship runtime_env.

These cover the control-plane decision of *how* batcher attaches to Ray and ships
its data plane to workers — pure kwargs construction, no Ray process required. The
flight-worker actors import `batcher` + its native extension to run; on a cluster
whose image doesn't already carry batcher (a source/editable install) that import
fails unless the package is uploaded via `runtime_env={"py_modules": [...]}`.
"""

from __future__ import annotations

import dataclasses
import os

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


_EXCLUDES = list(lifecycle._BUILD_ARTIFACT_EXCLUDES)


def test_self_ship_uploads_a_source_install(monkeypatch):
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    env = lifecycle._self_ship_runtime_env()
    # Ships the driver's package and pins pip off — workers use their base env for other
    # deps, and a managed hook can't inject an unresolvable per-job pip (e.g. batcher-engine).
    assert env == {
        "py_modules": ["/repo/python/batcher"],
        "pip": None,
        "excludes": _EXCLUDES,
    }


def test_self_ship_uploads_a_site_packages_install_too(monkeypatch):
    # A pip/site-packages install is shipped too: a driver that pip-installed batcher and
    # attached to an arbitrary cluster cannot assume that cluster's image carries a compatible
    # batcher, so shipping guarantees driver==worker code (the old skip-for-site-packages
    # heuristic produced a silent ModuleNotFoundError on the local-install → remote-cluster case).
    monkeypatch.setattr(
        lifecycle,
        "package_dir",
        lambda: "/opt/conda/lib/python3.12/site-packages/batcher",
    )
    assert lifecycle._self_ship_runtime_env() == {
        "py_modules": ["/opt/conda/lib/python3.12/site-packages/batcher"],
        "pip": None,
        "excludes": _EXCLUDES,
    }


def test_self_ship_skipped_only_when_cluster_image_is_trusted(monkeypatch, restore_config):
    # The one opt-out: a production image that bakes a matching batcher into every node. No
    # upload, but pip is still pinned off so a managed hook can't inject a broken per-job pip
    # — and build output is still excluded from the working_dir the platform injects.
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _with_distributed(trust_cluster_image=True)
    assert lifecycle._self_ship_runtime_env() == {"pip": None, "excludes": _EXCLUDES}


def test_self_ship_excludes_build_output_from_the_injected_working_dir(monkeypatch):
    """Batcher sets no `working_dir` — but a managed workspace injects one (the whole
    project dir), and Ray zips it on every `ray.init` under a hard 512 MiB cap. A project
    built in place carries its build output there: a `cargo` `target/` runs to gigabytes
    and fails `ray.init` outright ("Package size exceeds the maximum size of 512.00MiB")
    before any work starts — from a distributed query, for a reason unrelated to it.
    """
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    excludes = lifecycle._self_ship_runtime_env()["excludes"]
    for pattern in ("**/target/release/**", "**/target/debug/**", "**/docs/_build/**"):
        assert pattern in excludes
    # Source is never excluded — the upload exists to ship it.
    assert not any("python/batcher" in p for p in excludes)


def test_init_kwargs_attach_auto_ships_when_runtime_env_unset(monkeypatch, restore_config):
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _with_distributed(ray_address="auto", runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=4)
    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs  # never pin CPUs against a real cluster
    assert kwargs["runtime_env"] == {
        "py_modules": ["/repo/python/batcher"],
        "pip": None,
        "excludes": _EXCLUDES,
    }


def test_explicit_runtime_env_wins_over_auto_ship(monkeypatch, restore_config):
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    explicit = {"pip": ["numpy"]}
    _with_distributed(ray_address="auto", runtime_env=explicit)
    assert lifecycle._ray_init_kwargs(workers=4)["runtime_env"] == explicit


def test_local_cluster_does_not_auto_ship(_no_cluster_signal, restore_config):
    _no_cluster_signal.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _with_distributed(ray_address=None, runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=3)
    # Spinning a local in-process cluster: cap CPUs, ship nothing (workers are local).
    assert kwargs["num_cpus"] == 3
    assert "runtime_env" not in kwargs


def test_managed_cluster_attaches_without_ray_address(_no_cluster_signal, restore_config):
    # A managed workspace (Anyscale) that exports no RAY_ADDRESS must still ATTACH to the
    # running cluster — a bare local start would strand a distributed job on one node.
    _no_cluster_signal.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _no_cluster_signal.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _with_distributed(ray_address=None, runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=3)
    assert kwargs["address"] == "auto"
    assert "num_cpus" not in kwargs  # never pin CPUs against a real cluster
    assert kwargs["runtime_env"] == {
        "py_modules": ["/repo/python/batcher"],
        "pip": None,
        "excludes": _EXCLUDES,
    }


def test_ray_address_env_value_is_the_address_not_just_a_signal(_no_cluster_signal, monkeypatch):
    """`RAY_ADDRESS` names *which* cluster to attach to. Collapsing it to `"auto"` discards
    the only disambiguation Ray has when a host runs more than one instance (a managed
    cluster plus a stray local Ray from a colocated test run), and `ray.init(address="auto")`
    then dies with "Found multiple active Ray instances ... set the RAY_ADDRESS environment
    variable" — the very thing the user had already set.
    """
    monkeypatch.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    monkeypatch.setenv("RAY_ADDRESS", "10.0.3.113:6379")
    _with_distributed(ray_address=None, runtime_env=None)
    assert lifecycle._ray_init_kwargs(workers=4)["address"] == "10.0.3.113:6379"


def test_managed_cluster_without_ray_address_still_uses_auto(_no_cluster_signal, restore_config):
    _no_cluster_signal.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _no_cluster_signal.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _with_distributed(ray_address=None, runtime_env=None)
    assert lifecycle._ray_init_kwargs(workers=4)["address"] == "auto"


def test_platform_env_hook_is_disabled_across_our_ray_init(monkeypatch):
    """Ray applies `RAY_RUNTIME_ENV_HOOK` to the runtime_env *after* we build it, so a
    managed platform's hook rewrites the env `_self_ship_runtime_env` just constructed: it
    substitutes the workspace's tracked pip list for our `pip: None` (Ray's `RuntimeEnv`
    drops falsey values, so the hook sees no `pip` key) and zips the project dir as a
    `working_dir` before our `excludes` are ever applied. Both are fatal — an unresolvable
    requirement makes every worker's runtime-env build fail, and a `cargo target/` blows
    Ray's package cap. Neutralize the hook for the duration of our own `ray.init`.
    """
    monkeypatch.setenv("RAY_RUNTIME_ENV_HOOK", "platform_plugin._hook")
    with lifecycle._platform_env_hook_disabled():
        assert "RAY_RUNTIME_ENV_HOOK" not in os.environ
    # Scoped, not global: any other Ray user in the process still gets the platform's hook.
    assert os.environ["RAY_RUNTIME_ENV_HOOK"] == "platform_plugin._hook"


def test_platform_env_hook_restored_even_when_ray_init_raises(monkeypatch):
    monkeypatch.setenv("RAY_RUNTIME_ENV_HOOK", "platform_plugin._hook")
    with pytest.raises(RuntimeError), lifecycle._platform_env_hook_disabled():
        raise RuntimeError("ray.init blew up")
    assert os.environ["RAY_RUNTIME_ENV_HOOK"] == "platform_plugin._hook"


def test_platform_env_hook_absent_is_a_no_op(monkeypatch):
    monkeypatch.delenv("RAY_RUNTIME_ENV_HOOK", raising=False)
    with lifecycle._platform_env_hook_disabled():
        assert "RAY_RUNTIME_ENV_HOOK" not in os.environ
    assert "RAY_RUNTIME_ENV_HOOK" not in os.environ


def test_force_local_overrides_managed_detection(_no_cluster_signal, restore_config):
    # The reachability fallback: even on a managed workspace, `force_local` starts a local
    # single-node Ray (used when `address="auto"` found no reachable cluster).
    _no_cluster_signal.setattr(lifecycle, "package_dir", lambda: "/repo/python/batcher")
    _no_cluster_signal.setenv("ANYSCALE_SESSION_ID", "ses_abc")
    _with_distributed(ray_address="auto", runtime_env=None)
    kwargs = lifecycle._ray_init_kwargs(workers=5, force_local=True)
    assert "address" not in kwargs
    assert kwargs["num_cpus"] == 5
