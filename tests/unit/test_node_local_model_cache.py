"""Model weights belong on the node's disk, not on the container's overlay.

The default HuggingFace cache is under `$HOME`, which on a GPU node is a 20-100 GB overlay
shared with the image and every other tenant, while the node's terabytes of NVMe sit empty.
Eight workers per node each want the same tens of gigabytes there. The failure is `ENOSPC`
partway through a shard download, on one worker, after several minutes -- or a job that works
at four workers per node and fills the disk at eight.

Every test here is about the three ways the redirect would be *wrong*, because each of them
fails silently and looks like success: overriding an operator who named a cache, forcing a
re-download of models an image already baked in, and setting an environment variable after the
hub client has read it.
"""

from __future__ import annotations

import os
import sys

import pytest

from batcher._internal.site import model_cache

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No inherited cache variables, no real home directory, no memoized decision."""
    for name in model_cache.CACHE_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    model_cache.reset_model_cache_probe()
    yield
    model_cache.reset_model_cache_probe()


def _scratch(monkeypatch, path: str | None) -> None:
    monkeypatch.setattr(
        "batcher._internal.site.scratch.local_scratch_root", lambda: path, raising=True
    )


def test_the_cache_lands_on_the_measured_local_volume(monkeypatch, tmp_path) -> None:
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    expected = str(tmp_path / "nvme" / model_cache.MODEL_CACHE_DIRNAME)
    assert model_cache.model_cache_root() == expected
    assert model_cache.use_node_local_model_cache() == expected
    assert os.environ["HF_HUB_CACHE"] == expected
    assert os.environ["HF_HOME"] == expected
    assert os.path.isdir(expected)


@pytest.mark.parametrize("name", model_cache.CACHE_ENVS)
def test_an_operator_who_named_a_cache_is_not_overridden(monkeypatch, tmp_path, name) -> None:
    # A fleet mounting a shared network cache does it through exactly these variables.
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    monkeypatch.setenv(name, "/mnt/shared/models")
    assert model_cache.model_cache_root() is None
    assert model_cache.use_node_local_model_cache() is None


def test_an_image_that_already_baked_the_models_in_is_left_alone(monkeypatch, tmp_path) -> None:
    # Redirecting here does not save a download, it forces one -- of tens of gigabytes, on
    # every worker, for a model the node already had.
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    hub = tmp_path / "home" / ".cache" / "huggingface" / "hub" / "models--org--model"
    hub.mkdir(parents=True)
    assert model_cache.model_cache_root() is None


def test_a_relocated_default_cache_is_found_too(monkeypatch, tmp_path) -> None:
    # `XDG_CACHE_HOME` moves the whole default. Missing it would redirect past a baked-in
    # model and force exactly the download this check exists to prevent.
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    (xdg / "huggingface" / "hub" / "models--org--model").mkdir(parents=True)
    assert model_cache.model_cache_root() is None


def test_an_unrelated_directory_in_the_default_cache_is_not_a_baked_model(
    monkeypatch, tmp_path
) -> None:
    # `.locks`, `xet`, and a stray file are not weights, so they must not veto the redirect.
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    hub = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    hub.mkdir(parents=True)
    (hub / ".locks").mkdir()
    assert model_cache.model_cache_root() is not None


def test_a_node_with_no_local_volume_keeps_the_default(monkeypatch) -> None:
    _scratch(monkeypatch, None)
    assert model_cache.model_cache_root() is None
    assert model_cache.use_node_local_model_cache() is None
    assert "HF_HUB_CACHE" not in os.environ


def test_it_refuses_once_the_hub_client_has_read_its_cache_path(monkeypatch, tmp_path) -> None:
    # The variable is read into module constants at import. Setting it afterwards changes
    # nothing, and reporting success would be a lie that looks exactly like the truth.
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", object())
    assert model_cache.use_node_local_model_cache() is None
    assert "HF_HUB_CACHE" not in os.environ


def test_an_uncreatable_directory_is_not_reported_as_in_force(monkeypatch, tmp_path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    _scratch(monkeypatch, str(blocker))
    assert model_cache.use_node_local_model_cache() is None
    assert "HF_HUB_CACHE" not in os.environ


def test_the_decision_is_made_once_per_process(monkeypatch, tmp_path) -> None:
    _scratch(monkeypatch, str(tmp_path / "nvme"))
    first = model_cache.use_node_local_model_cache()
    _scratch(monkeypatch, str(tmp_path / "other"))
    assert model_cache.use_node_local_model_cache() == first


def test_building_a_class_udf_chooses_the_cache_first(monkeypatch, tmp_path) -> None:
    """The build hook is the last point before a model loads that Batcher still controls."""
    from batcher.core.udf.lifecycle import build_udf_callable

    _scratch(monkeypatch, str(tmp_path / "nvme"))

    class _Model:
        def __call__(self, batch):  # pragma: no cover - never invoked here
            return batch

    build_udf_callable(_Model)
    assert os.environ["HF_HUB_CACHE"].endswith(model_cache.MODEL_CACHE_DIRNAME)


def test_a_plain_callable_udf_touches_nothing(monkeypatch, tmp_path) -> None:
    from batcher.core.udf.lifecycle import build_udf_callable

    _scratch(monkeypatch, str(tmp_path / "nvme"))
    assert build_udf_callable(len) is len
    assert "HF_HUB_CACHE" not in os.environ
