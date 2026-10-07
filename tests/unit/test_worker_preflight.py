"""The worker binary-compatibility preflight: comparison, enforcement, and caching.

Every probe result here is fabricated, so these run with no cluster and no Ray. The fake
`ray` module answers only what `ensure_workers_compatible` asks of it (initialized, the
driver's node id, the node list); the probe itself is replaced, which is the seam the real
fan-out sits behind. `tests/integration/test_worker_preflight_ray.py` runs the real probe.
"""

from __future__ import annotations

import dataclasses
import logging
import types

import pytest

from batcher._internal.errors import BackendError
from batcher.config import active_config, set_config
from batcher.dist.executors.ray_runtime.preflight import report, run
from batcher.dist.executors.ray_runtime.preflight.report import PlatformFacts, compare

pytestmark = pytest.mark.unit

_PKGS = {"pyarrow": "17.0.0", "numpy": "1.26.4", "psutil": "5.9.8"}
X86 = PlatformFacts("linux", "x86_64", "glibc", "2.35", "3.12", "cpython", dict(_PKGS))
ARM = dataclasses.replace(X86, machine="aarch64")


def _raw(facts: PlatformFacts) -> dict:
    return {**dataclasses.asdict(facts), "packages": dict(facts.packages)}


class _FakeRay(types.SimpleNamespace):
    """The slice of `ray` the preflight reads: is it up, where am I, which nodes exist."""

    def __init__(self, node_ids: list[str], here: str = "driver") -> None:
        super().__init__()
        self._nodes = [
            {"NodeID": nid, "Alive": True, "NodeManagerAddress": f"10.0.0.{i}"}
            for i, nid in enumerate([here, *node_ids])
        ]
        self._here = here

    def is_initialized(self) -> bool:
        return True

    def get_runtime_context(self):
        return types.SimpleNamespace(get_node_id=lambda: self._here)

    def nodes(self) -> list[dict]:
        return self._nodes


@pytest.fixture
def preflight(monkeypatch):
    """Fabricate the driver and each worker's answer; record which nodes were probed."""
    saved = active_config()
    run.reset_preflight_cache()
    probed: list[list[str]] = []
    answers: dict[str, PlatformFacts] = {}

    def fake_probe(ray, nodes, packages):
        ids = [n["NodeID"] for n in nodes]
        probed.append(ids)
        return {nid: _raw(answers[nid]) for nid in ids if nid in answers}, [
            nid for nid in ids if nid not in answers
        ]

    monkeypatch.setattr(run, "probe_nodes", fake_probe)
    monkeypatch.setattr(run, "_driver_facts", lambda: X86)
    monkeypatch.setattr(run, "_engine_floor", lambda: (2, 35))
    monkeypatch.setattr(run, "job_ships_batcher", lambda: True)
    yield types.SimpleNamespace(answers=answers, probed=probed)
    run.reset_preflight_cache()
    set_config(saved)


def _distributed(**overrides) -> None:
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **overrides)))


# --- the comparison ------------------------------------------------------------------------


def test_an_x86_driver_and_an_arm_worker_differ_on_architecture_and_it_blocks():
    found = compare(X86, {"n1": ("node arm-1", ARM)}, ships_driver_build=True)
    assert [(f.field, f.driver, f.worker, f.blocking) for f in found] == [
        ("architecture", "x86_64", "aarch64", True)
    ]
    assert "node arm-1: architecture is 'aarch64'" in found[0].render()


def test_matching_nodes_produce_no_findings():
    assert compare(X86, {"n1": ("n1", X86)}, ships_driver_build=True, glibc_floor=(2, 35)) == ()


def test_architecture_spellings_are_normalized_before_comparison():
    worker = PlatformFacts.from_probe({**_raw(X86), "machine": "AMD64"})
    assert worker.machine == "x86_64"
    assert compare(X86, {"n1": ("n1", worker)}, ships_driver_build=True) == ()


def test_a_trusted_image_reports_the_same_differences_without_blocking():
    found = compare(X86, {"n1": ("n1", ARM)}, ships_driver_build=False)
    assert [(f.field, f.blocking) for f in found] == [("architecture", False)]


@pytest.mark.parametrize(
    ("worker_glibc", "blocking"),
    [("2.31", True), ("2.35", None), ("2.39", None)],
)
def test_glibc_is_judged_against_the_engine_floor_not_the_drivers_version(worker_glibc, blocking):
    driver = dataclasses.replace(X86, libc_version="2.39")  # newer than the engine needs
    worker = dataclasses.replace(X86, libc_version=worker_glibc)
    found = compare(driver, {"n": ("n", worker)}, ships_driver_build=True, glibc_floor=(2, 35))
    assert [f.blocking for f in found] == ([] if blocking is None else [blocking])


def test_musl_against_glibc_blocks_on_the_c_library():
    musl = dataclasses.replace(X86, libc="", libc_version="")
    found = compare(X86, {"n": ("n", musl)}, ships_driver_build=True, glibc_floor=(2, 35))
    assert [(f.field, f.worker, f.blocking) for f in found] == [("C library", "not glibc", True)]


def test_python_minor_mismatch_blocks_when_the_driver_build_is_shipped():
    old = dataclasses.replace(X86, python="3.11")
    found = compare(X86, {"n": ("n", old)}, ships_driver_build=True)
    assert [(f.field, f.blocking) for f in found] == [("Python version", True)]


def test_missing_pyarrow_blocks_but_a_missing_optional_dep_or_other_version_only_reports():
    worker = dataclasses.replace(X86, packages={"pyarrow": "", "numpy": "2.0.0", "psutil": ""})
    found = {
        f.field: f.blocking for f in compare(X86, {"n": ("n", worker)}, ships_driver_build=True)
    }
    assert found == {"package pyarrow": True, "package numpy": False, "package psutil": False}


def test_the_workers_own_batcher_version_matters_only_on_a_trusted_image():
    driver = dataclasses.replace(X86, packages={report.ENGINE_DIST: "2.1.0"})
    worker = dataclasses.replace(X86, packages={report.ENGINE_DIST: "2.0.0"})
    assert compare(driver, {"n": ("n", worker)}, ships_driver_build=True) == ()
    trusted = compare(driver, {"n": ("n", worker)}, ships_driver_build=False)
    assert [(f.field, f.blocking) for f in trusted] == [("package batcher-engine", False)]


def test_the_probe_body_reports_this_interpreter_from_the_standard_library():
    import platform
    import sys

    raw = report.facts_on_this_node(("pyarrow", "surely-not-an-installed-dist"))
    assert raw["os"] == sys.platform
    assert raw["machine"] == platform.machine()
    assert raw["python"] == f"{sys.version_info[0]}.{sys.version_info[1]}"
    assert raw["packages"]["pyarrow"]
    assert raw["packages"]["surely-not-an-installed-dist"] == ""


# --- the glibc floor -----------------------------------------------------------------------


def test_the_engine_floor_is_the_highest_glibc_symbol_version_referenced(tmp_path):
    so = tmp_path / "_native.abi3.so"
    so.write_bytes(b"\x7fELF" + b"\0" * 64 + b"GLIBC_2.17\0GLIBC_2.2.5\0GLIBC_2.34\0GLIBC_2.9\0")
    assert report.engine_glibc_floor(so) == (2, 34)


def test_a_non_elf_file_or_a_missing_one_has_no_floor(tmp_path):
    other = tmp_path / "x.so"
    other.write_bytes(b"MZ GLIBC_2.40")
    assert report.engine_glibc_floor(other) is None
    assert report.engine_glibc_floor(tmp_path / "absent.so") is None


def test_the_installed_engine_has_a_readable_floor():
    """A positive control for the scan: the real extension names a glibc floor on Linux."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("glibc floors exist only for a Linux build")
    run._engine_floor.cache_clear()
    floor = run._engine_floor()
    assert floor is not None and floor >= (2, 17)


# --- enforcement and caching ---------------------------------------------------------------


def test_an_arm_worker_refuses_the_query_naming_the_node_and_field(preflight):
    preflight.answers["arm-node"] = ARM
    with pytest.raises(BackendError) as info:
        run.ensure_workers_compatible(_FakeRay(["arm-node"]))
    text = str(info.value)
    assert "node arm-node" in text
    assert "architecture is 'aarch64'" in text
    assert "x86_64" in text
    assert "trust_cluster_image" in info.value.hint
    assert preflight.probed == [["arm-node"]]  # the driver's own node is never probed


def test_matching_workers_pass_and_are_probed_once_per_session(preflight):
    preflight.answers.update({"w1": X86, "w2": X86})
    ray = _FakeRay(["w1", "w2"])
    first = run.ensure_workers_compatible(ray)
    second = run.ensure_workers_compatible(ray)
    assert first is not None and first.findings == () and len(first.probed) == 2
    assert second is not None and second.findings == ()
    assert preflight.probed == [["w1", "w2"]]


def test_a_refusal_is_repeated_from_the_cache_without_probing_again(preflight):
    preflight.answers["arm-node"] = ARM
    ray = _FakeRay(["arm-node"])
    for _ in range(2):
        with pytest.raises(BackendError):
            run.ensure_workers_compatible(ray)
    assert preflight.probed == [["arm-node"]]


def test_a_node_added_later_is_probed_alone(preflight, monkeypatch):
    preflight.answers.update({"w1": X86, "w2": ARM})
    run.ensure_workers_compatible(_FakeRay(["w1"]))
    monkeypatch.setattr(run, "_RECHECK_S", 0.0)
    with pytest.raises(BackendError, match="node w2"):
        run.ensure_workers_compatible(_FakeRay(["w1", "w2"]))
    assert preflight.probed == [["w1"], ["w2"]]


def test_a_trusted_image_warns_instead_of_refusing(preflight, caplog):
    _distributed(trust_cluster_image=True)
    preflight.answers["arm-node"] = ARM
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        got = run.ensure_workers_compatible(_FakeRay(["arm-node"]))
    assert got is not None and not got.ships_driver_build
    assert [f.field for f in got.findings] == ["architecture"]
    assert "architecture is 'aarch64'" in caplog.text


def test_an_explicit_runtime_env_is_the_users_contract_and_only_warns(preflight):
    _distributed(runtime_env={"pip": ["batcher-engine"]})
    preflight.answers["arm-node"] = ARM
    got = run.ensure_workers_compatible(_FakeRay(["arm-node"]))
    assert got is not None and not got.ships_driver_build
    assert [f.field for f in got.findings] == ["architecture"] and got.blocking == ()


def test_a_foreign_ray_init_ships_per_remote_and_is_enforced(preflight, monkeypatch):
    _distributed(runtime_env={"pip": ["x"]})  # ignored: Batcher did not run this ray.init
    monkeypatch.setattr(run, "job_ships_batcher", lambda: False)
    preflight.answers["arm-node"] = ARM
    with pytest.raises(BackendError):
        run.ensure_workers_compatible(_FakeRay(["arm-node"]))


def test_a_silent_node_is_reported_unverified_and_not_refused(preflight, caplog):
    with caplog.at_level(logging.WARNING, logger="batcher.dist"):
        got = run.ensure_workers_compatible(_FakeRay(["slow-node"]))
    assert got is not None and got.unanswered == ("node slow-node (10.0.0.1)",)
    assert "not verified" in caplog.text


def test_failing_to_ask_never_stops_a_query(preflight):
    class Broken(_FakeRay):
        def nodes(self):
            raise RuntimeError("GCS unreachable")

    assert run.ensure_workers_compatible(Broken(["w"])) is None


def test_a_local_cluster_probes_nothing(preflight):
    got = run.ensure_workers_compatible(_FakeRay([]))
    assert got is not None and got.findings == () and preflight.probed == []


# --- the probe ships without Batcher -------------------------------------------------------


def test_the_shipped_probe_pickles_by_value_with_no_reference_to_batcher():
    """A by-reference pickle would make the worker `import batcher`, and so the engine."""
    import subprocess
    import sys

    cloudpickle = pytest.importorskip("cloudpickle")
    shipped = cloudpickle.dumps(run._by_value(report.facts_on_this_node))
    by_reference = cloudpickle.dumps(report.facts_on_this_node)
    # Unpickle and call in an interpreter where importing `batcher` raises: what a worker
    # that cannot load the engine would experience. The module-level function is the
    # positive control, proving the guard does fire on a by-reference pickle.
    child = (
        "import sys, cloudpickle\n"
        "class Deny:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'batcher':\n"
        "            raise ImportError('batcher imported on the worker')\n"
        "sys.meta_path.insert(0, Deny())\n"
        "fn = cloudpickle.loads(sys.stdin.buffer.read())\n"
        "print(fn(('pyarrow',))['os'])\n"
    )

    def load(payload: bytes) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-I", "-c", child], input=payload, capture_output=True, check=False
        )

    ok = load(shipped)
    assert ok.returncode == 0, ok.stderr.decode()
    assert ok.stdout.decode().strip() == sys.platform
    denied = load(by_reference)
    assert denied.returncode != 0 and b"batcher imported on the worker" in denied.stderr
