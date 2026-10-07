"""`ds.explain(requirements=True)`: what a plan's remote workers must import (AP-038).

The acceptance is that a missing user module is diagnosed before model workers start. A
module written to a temporary directory stands in for the user's ``models.py``: importable
on the driver, provided by no installed distribution, so a worker has it only if shipped.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _report(ds: bt.Dataset) -> dict:
    return json.loads(ds.explain(format="json", requirements=True))["requirements"]


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    (tmp_path / "xai_user_models.py").write_text("def ident(batch):\n    return batch\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path / "xai_user_models.py"
    sys.modules.pop("xai_user_models", None)


def test_a_udf_from_a_local_module_is_flagged_before_anything_runs(user_module):
    from xai_user_models import ident

    ds = bt.from_pydict({"x": [1]}).map_batches(ident)
    report = _report(ds)
    assert report["status"] == "warn"
    (stage,) = report["stages"]
    assert stage["stage"] == "ident"
    assert stage["modules"] == [
        {
            "module": "xai_user_models",
            "kind": "local",
            "how": "reference",
            "path": str(user_module),
            "covered": None,
        }
    ]
    text = ds.explain(requirements=True)
    assert "xai_user_models [local:" in text
    assert "py_modules" in text


def test_a_local_module_shipped_by_the_runtime_env_is_covered(user_module, monkeypatch):
    ray = pytest.importorskip("ray")
    from xai_user_models import ident

    class _Context:
        def __init__(self) -> None:
            self.runtime_env = {"py_modules": [str(user_module.parent)]}

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: _Context())
    report = _report(bt.from_pydict({"x": [1]}).map_batches(ident))
    assert report["stages"][0]["modules"][0]["covered"] is True
    assert report["status"] == "ok"


def test_imports_made_inside_the_fn_are_reported_and_packages_carry_versions():
    def needs_more(batch):
        import numpy  # noqa: F401  (resolved on the worker, at call time)
        import xai_definitely_missing_pkg  # noqa: F401

        return batch

    report = _report(bt.from_pydict({"x": [1]}).map_batches(needs_more))
    modules = {m["module"]: m for m in report["stages"][0]["modules"]}
    assert modules["numpy"]["kind"] == "package"
    assert modules["numpy"]["version"]
    assert modules["numpy"]["how"] == "import"
    assert modules["xai_definitely_missing_pkg"]["kind"] == "missing"
    assert report["status"] == "warn"
    assert "json" not in modules  # the standard library is never listed


def test_an_unpicklable_closure_is_an_error_naming_the_variable():
    lock = threading.Lock()

    def holds_a_lock(batch):
        with lock:
            return batch

    report = _report(bt.from_pydict({"x": [1]}).map_batches(holds_a_lock))
    assert report["status"] == "error"
    (stage,) = report["stages"]
    assert stage["picklable"] is False
    assert "lock" in stage["problems"][0]


def test_a_model_engine_factory_import_is_seen_through_the_class_udf():
    """`vllm_engine` imports vLLM inside its factory, so the driver never needs it."""
    ds = bt.from_pydict({"q": ["hi"]})
    report = _report(ds.ml.generate(bt.ml.vllm_engine("some/model"), prompt_column="q"))
    names = {m["module"] for m in report["stages"][0]["modules"]}
    assert "vllm" in names


def test_a_plan_without_udfs_needs_only_the_engine():
    ds = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 0)
    assert _report(ds) == {"status": "ok", "stages": []}
    assert ds.explain(requirements=True).rstrip().endswith("workers need only the engine")


def test_the_plain_explain_is_unchanged():
    ds = bt.from_pydict({"x": [1]}).map_batches(lambda b: b)
    # Positive control first: the same plan renders both tokens when asked.
    assert "worker requirements" in ds.explain(requirements=True)
    assert "requirements" in json.loads(ds.explain(format="json", requirements=True))
    assert "worker requirements" not in ds.explain()
    assert "requirements" not in json.loads(ds.explain(format="json"))
