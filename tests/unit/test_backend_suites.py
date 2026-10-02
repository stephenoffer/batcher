"""The backend lanes' file selector must find every spelling of "this test needs Ray".

`ci.yml`'s `distributed` job is the only place a Ray-dependent test runs: the `gate` job has
no Ray, so there they skip. The job used to select files with
``grep -rl 'importorskip("ray"'``, and every shape below except the first slipped past it —
each one skipped in `gate`, was not selected by the Ray lane, and therefore ran nowhere.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools import backend_suites

pytestmark = pytest.mark.unit


def _tree(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    tests = tmp_path / "tests"
    for rel, body in files.items():
        path = tests / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tests


def _selected(tmp_path: pathlib.Path, files: dict[str, str]) -> list[str]:
    return backend_suites.discover(_tree(tmp_path, files))


@pytest.mark.parametrize(
    "body",
    [
        'import pytest\npytest.importorskip("ray")\n',
        "import pytest\npytest.importorskip('ray')\n",
        'import pytest\nserve = pytest.importorskip("ray.serve")\n',
        "def test_x():\n    import ray\n    assert ray\n",
        "def test_x():\n    from ray import serve\n    assert serve\n",
        'import importlib.util\nHAS = importlib.util.find_spec("ray") is not None\n',
        "def test_x(ds):\n    assert ds.collect(distributed=True)\n",
    ],
    ids=[
        "double-quoted",
        "single-quoted",
        "submodule",
        "body-import",
        "from-import",
        "find-spec",
        "distributed-true",
    ],
)
def test_each_spelling_of_a_ray_dependency_is_selected(tmp_path, body):
    assert _selected(tmp_path, {"integration/test_subject.py": body}) == [
        "tests/integration/test_subject.py"
    ]


def test_a_conftest_gate_selects_its_whole_subtree(tmp_path):
    selected = _selected(
        tmp_path,
        {
            "dist/conftest.py": "import pytest\npytest.importorskip('ray')\n",
            "dist/test_a.py": "def test_a():\n    assert 1\n",
            "dist/deep/test_b.py": "def test_b():\n    assert 1\n",
            "other/test_c.py": "def test_c():\n    assert 1\n",
        },
    )
    assert selected == ["tests/dist/deep/test_b.py", "tests/dist/test_a.py"]


def test_a_shared_ray_helper_selects_the_files_importing_it(tmp_path):
    selected = _selected(
        tmp_path,
        {
            "_ray_cluster.py": "def start():\n    import ray\n    return ray\n",
            "_plain.py": "X = 1\n",
            "integration/test_uses_helper.py": "from _ray_cluster import start\n",
            "integration/test_uses_plain.py": "from _plain import X\n",
        },
    )
    assert selected == ["tests/integration/test_uses_helper.py"]


@pytest.mark.parametrize(
    "body",
    [
        "def test_x(ds):\n    assert ds.collect(distributed=False)\n",
        'def test_x():\n    assert "ray" in "array"\n',
        "import rayon_like\n",
        'import pytest\npytest.importorskip("rayon")\n',
    ],
    ids=["distributed-false", "string-mention", "prefix-module", "prefix-gate"],
)
def test_a_file_that_does_not_need_ray_is_not_selected(tmp_path, body):
    assert _selected(tmp_path, {"unit/test_subject.py": body}) == []


def test_a_suite_that_stops_being_detected_is_named(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_still_here.py").write_text("")
    problems = backend_suites.check(
        selected=["tests/test_new.py"],
        inventory=["tests/test_still_here.py", "tests/test_gone.py"],
        base=tmp_path,
    )
    joined = "\n".join(problems)
    assert "test_still_here.py: still exists but is no longer detected as a ray suite" in joined
    assert "test_gone.py: on the inventory but deleted or renamed" in joined
    assert "test_new.py: a ray suite not on the inventory" in joined


def test_a_selection_matching_its_inventory_has_no_problems(tmp_path):
    files = ["tests/test_a.py", "tests/test_b.py"]
    assert backend_suites.check(selected=files, inventory=files, base=tmp_path) == []


def test_shards_partition_the_selection():
    files = [f"tests/test_{i}.py" for i in range(10)]
    shards = [backend_suites.shard(files, i, 4) for i in range(1, 5)]
    assert sorted(f for s in shards for f in s) == sorted(files)
    assert all(shards)


def test_the_real_tree_selects_a_suite_the_old_grep_missed():
    # Gated with `@pytest.mark.skipif` and a `from ray import serve` inside a helper, never
    # with `importorskip("ray"` — so the grep selector never ran it.
    selected = backend_suites.discover()
    subject = backend_suites.ROOT / "tests" / "integration" / "test_online_serving.py"
    assert "tests/integration/test_online_serving.py" in selected
    assert 'importorskip("ray"' not in subject.read_text()


def test_a_framework_selection_ignores_distributed_true(tmp_path):
    files = {
        "unit/test_torch.py": "import pytest\ntorch = pytest.importorskip('torch')\n",
        "unit/test_cluster.py": "def test_x(ds):\n    assert ds.collect(distributed=True)\n",
    }
    tests = _tree(tmp_path, files)
    assert backend_suites.discover(tests, dep="torch") == ["tests/unit/test_torch.py"]
    assert backend_suites.discover(tests, dep="ray") == ["tests/unit/test_cluster.py"]
