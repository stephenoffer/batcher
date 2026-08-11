"""The allowlist staleness audit must judge an entry against the limit it actually exempts.

`tools/lint_structure.py` reports an exemption whose file no longer needs it, so the list
shrinks over time instead of only growing. The audit is advisory, which is exactly what
makes a false positive expensive: it reads as "delete this", and deleting a *live*
exemption fails the very gate the entry was suppressing.

That happened. The audit tested every Python entry against `PY_HARD` (500), but an
`__init__.py` is allowlisted against `INIT_MAX` (120) — the re-export ceiling. A 120-line
ceiling guarantees the file is under 500, so all seven `__init__.py` exemptions were
reported deletable while every one of them was load-bearing. The second half of the same
bug was the metric: the audit counted raw lines where `check_python_file` counts code
lines excluding docstrings, so the two halves of the checker disagreed about a file's size.

These tests pin both halves.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "lint_structure.py"


def _load():
    """Import `tools/lint_structure.py` by path — `tools/` is a script dir, not a package."""
    spec = importlib.util.spec_from_file_location("_lint_structure_under_test", _TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stale_for(monkeypatch, path: pathlib.Path) -> list[str]:
    """Run the audit over an allowlist holding exactly `path`; return its findings."""
    mod = _load()
    monkeypatch.setattr(mod, "STRUCTURE_ALLOW", {str(path): "under test"})
    monkeypatch.setattr(mod, "DIR_ALLOW", {})
    return mod.stale_allowlist_entries()


def _write(tmp_path: pathlib.Path, name: str, code_lines: int) -> pathlib.Path:
    """Write a syntactically valid module of exactly `code_lines` countable lines."""
    path = tmp_path / name
    path.write_text("".join(f"x{i} = {i}\n" for i in range(code_lines)))
    return path


# --- the regression: an __init__.py is judged against the re-export ceiling -----------


def test_live_init_exemption_is_not_reported_stale(tmp_path, monkeypatch):
    """An `__init__.py` over 120 code lines still needs its entry, even though it is under 500."""
    init = _write(tmp_path, "__init__.py", 200)
    assert _stale_for(monkeypatch, init) == []


def test_init_exemption_below_the_re_export_ceiling_is_reported_stale(tmp_path, monkeypatch):
    """Once it drops under 120 the entry is genuinely dead and must be reported."""
    init = _write(tmp_path, "__init__.py", 40)
    (finding,) = _stale_for(monkeypatch, init)
    assert "120" in finding


# --- the ordinary module ceiling is unchanged ----------------------------------------


def test_module_under_the_python_ceiling_is_reported_stale(tmp_path, monkeypatch):
    module = _write(tmp_path, "small.py", 100)
    (finding,) = _stale_for(monkeypatch, module)
    assert "500" in finding


def test_module_over_the_python_ceiling_is_not_reported_stale(tmp_path, monkeypatch):
    module = _write(tmp_path, "big.py", 600)
    assert _stale_for(monkeypatch, module) == []


# --- the audit counts code lines, the same metric the gate counts ---------------------


def test_docstring_lines_do_not_keep_a_dead_exemption_alive(tmp_path, monkeypatch):
    """A file that is long only in docstrings is under the real limit, so its entry is stale."""
    module = tmp_path / "documented.py"
    body = "".join(f"x{i} = {i}\n" for i in range(100))
    filler = "filler line\n" * 600
    module.write_text('"""Doc.\n\n' + filler + '"""\n' + body)
    (finding,) = _stale_for(monkeypatch, module)
    assert "500" in finding


def test_a_missing_path_is_still_reported(tmp_path, monkeypatch):
    assert _stale_for(monkeypatch, tmp_path / "gone.py") == [
        f"{tmp_path / 'gone.py'}: no such file"
    ]
