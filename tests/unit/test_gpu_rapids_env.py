"""Delivering cuDF to the workers: the shared mount, the derived pin, and the fallback order.

The device tier is unreachable on a fleet whose image ships no RAPIDS, and there are two ways
to reach it that differ by two orders of magnitude in cost. Measured on a six-T4 fleet: a Ray
`pip` block costs **168 s per node** to resolve, and a `PYTHONPATH` entry pointing at a staged
tree on a shared mount costs **9.1 s once** for the first cold NFS import. These tests pin the
three decisions that follow: that the mount wins when it is there, that the pip fallback names
the *driver's* cuDF rather than a hardcoded release, and that a configured-but-absent directory
is ignored rather than trusted.

That last one is the failure mode with no symptom: a `PYTHONPATH` pointing at nothing gives the
workers no cuDF *and* suppresses the pip block that would have, so every GPU task fails to
import and the tier silently disappears behind the CPU fallback.
"""

from __future__ import annotations

import pytest

from batcher.config import Config, DistributedConfig, active_config, config_context
from batcher.dist.gpu.cudf_probe import (
    _cuda_wheel_suffix,
    cudf_pip_spec,
    rapids_env_path,
    stage_rapids_env,
)

pytestmark = pytest.mark.unit


def _with_path(path: str):
    """A config context with `gpu_rapids_path` set and everything else left alone."""
    dc = active_config().distributed
    return config_context(Config().replace(distributed=_replaced(dc, path)))


def _replaced(dc: DistributedConfig, path: str) -> DistributedConfig:
    import dataclasses

    return dataclasses.replace(dc, gpu_rapids_path=path)


# --- the mount ---------------------------------------------------------------


def test_no_configured_path_means_the_mechanism_is_off():
    with _with_path(""):
        assert rapids_env_path() == ""


def test_a_configured_directory_that_exists_is_used(tmp_path):
    with _with_path(str(tmp_path)):
        assert rapids_env_path() == str(tmp_path)


def test_a_configured_directory_that_does_not_exist_is_ignored(tmp_path):
    """The one that has no symptom: a PYTHONPATH into nothing gives no cuDF *and* no pip block."""
    missing = tmp_path / "not-there"
    with _with_path(str(missing)):
        assert rapids_env_path() == ""


def test_a_file_where_a_directory_was_configured_is_ignored(tmp_path):
    stray = tmp_path / "rapids"
    stray.write_text("")
    with _with_path(str(stray)):
        assert rapids_env_path() == ""


# --- the pip fallback --------------------------------------------------------


def test_the_pip_spec_names_the_drivers_own_cudf():
    """A hardcoded `cudf-cu13==26.6.0` is wrong on every fleet that is not the one it was
    written on: a CUDA-12 image needs `cudf-cu12`, and a driver on another RAPIDS release ships
    partials the workers cannot unpickle."""
    spec = cudf_pip_spec()
    if not spec:  # no cuDF on this machine; there is nothing that could be pinned
        pytest.skip("the driver has no cuDF to describe")
    (requirement,) = spec
    import cudf

    name, _, version = requirement.partition("==")
    assert name == f"cudf-{_cuda_wheel_suffix()}"
    # `26.06.00` from cuDF, `26.6.0` on PyPI. An unnormalized pin resolves to nothing, which
    # turns a slow environment build into a failed one.
    assert version == ".".join(str(int(p)) for p in cudf.__version__.split(".")[:3])


def test_the_pip_spec_does_not_pin_numpy():
    """It used to pin `numpy==1.26.4`, and that pin made the failure it was written to prevent.

    RAPIDS 25.04 and later support numpy 2, so on a numpy-2 driver the pin dragged the workers
    *back* to numpy 1 — and Ray pickles arrays by module path, so every array a task returned
    then failed to unpickle on the driver with `No module named 'numpy._core'`.
    """
    assert not [r for r in cudf_pip_spec() if r.lower().startswith("numpy")]


# --- staging -----------------------------------------------------------------


def test_staging_is_idempotent_when_the_tree_is_already_there(tmp_path):
    """Called before every query, so an already-populated directory must cost one `isdir`."""
    (tmp_path / "cudf").mkdir()
    assert stage_rapids_env(str(tmp_path)) == str(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["cudf"]


def test_staging_nowhere_is_a_no_op():
    with _with_path(""):
        assert stage_rapids_env() == ""
