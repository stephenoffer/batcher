"""The audit detectors must not report what a reader would then have to dismiss.

`tools/audit_health.py` is a triage list, and a triage list is only worth reading while its
precision holds. Each test here pins one false positive the detectors actually produced on
this tree, so the calibration cannot silently regress:

* the reference index skipped every path with a `_build` component, which is the real package
  `api/dataset/_build/` as well as Sphinx output — so a name referenced only from those three
  modules read as dead, and `_all_bounded` was reported while `sessions.py` imports it;
* `SILENT_ALLOW` was keyed by line number and had already drifted 23 lines, so a waived
  handler came back as a `high` finding and the waiver became invisible debt;
* a base-class hook with a trivial body and a real subclass override read as an unkept
  promise, which flagged `CountVectorizer._weights` for being annotated `-> Any` rather than
  `-> X | None` while its sibling hooks were excused.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.audit import context as ctx_mod
from tools.audit import silent
from tools.audit.context import Context, build_context

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tree_context() -> Context:
    """The real tree, parsed once — the index bug is only visible against real paths."""
    return build_context()


def _synthetic(source: str, rel: str = "python/batcher/subject.py") -> Context:
    """A Context over one synthetic module, for the rules that need no cross-file index."""
    path = ctx_mod.ROOT / rel
    return Context(
        modules={path: ast.parse(source)},
        sources={path: source},
        name_files={},
        rust_text={},
    )


# --- the reference index ------------------------------------------------------------


def test_the_index_covers_the_real_build_package(tree_context: Context) -> None:
    """`api/dataset/_build/` is source, not build output, so its references must count."""
    seen = set(tree_context.name_files.get("_all_bounded", {}))
    assert "python/batcher/api/dataset/_build/sessions.py" in seen


def test_a_helper_imported_only_within_that_package_is_not_called_dead(
    tree_context: Context,
) -> None:
    """The finding this produced: `sessions.py` imports `_all_bounded` from `core.py`."""
    referenced = tree_context.used_outside(
        "_all_bounded", "python/batcher/api/dataset/_build/core.py"
    )
    assert referenced, "a name imported by a sibling module is referenced, not dead"


def test_the_docs_build_output_is_still_excluded() -> None:
    """Narrowing the exclusion must not stop it excluding what it was written for."""
    assert "docs/_build".startswith(ctx_mod.BUILD_OUTPUT_PREFIXES)
    assert not "python/batcher/api/dataset/_build/core.py".startswith(ctx_mod.BUILD_OUTPUT_PREFIXES)


# --- the silent-handler ledger ------------------------------------------------------


def test_every_shipped_waiver_still_resolves(tree_context: Context) -> None:
    """A ledger entry that matches nothing is the drift, so the shipped ledger must be clean."""
    stale = [f for f in silent.detect_swallowed(tree_context) if f.category == "stale-waiver"]
    assert stale == [], f"stale SILENT_ALLOW entries: {[f.message for f in stale]}"


def test_a_waiver_that_no_longer_resolves_is_reported(
    tree_context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drift must cost a line of output, not a silently unprotected site."""
    monkeypatch.setattr(ctx_mod, "SILENT_ALLOW", {"python/batcher/x.py::gone": "moved"})
    monkeypatch.setattr(silent, "SILENT_ALLOW", {"python/batcher/x.py::gone": "moved"})
    stale = [f for f in silent.detect_swallowed(tree_context) if f.category == "stale-waiver"]
    assert [f.severity for f in stale] == ["high"]


def test_the_retired_line_numbered_key_form_is_reported_too(
    tree_context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact key that drifted: it must not read as a resolving waiver."""
    old = {"python/batcher/dist/executors/scan_read.py:171": "the drifted key"}
    monkeypatch.setattr(ctx_mod, "SILENT_ALLOW", old)
    monkeypatch.setattr(silent, "SILENT_ALLOW", old)
    found = list(silent.detect_swallowed(tree_context))
    assert any(f.category == "stale-waiver" for f in found)
    # ...and the site it used to cover is reported rather than quietly waived.
    assert any(
        f.category == "swallowed-error" and "scan_read.py" in f.path and f.severity == "high"
        for f in found
    )


def test_the_logging_path_handler_stays_waived_by_symbol(tree_context: Context) -> None:
    """`_record_skipped` cannot trace its own failure — the one site that must stay silent."""
    reported = [
        f
        for f in silent.detect_swallowed(tree_context)
        if f.category == "swallowed-error" and f.severity == "high" and "scan_read.py" in f.path
    ]
    assert reported == []


# --- stubs versus hooks -------------------------------------------------------------


def test_a_base_hook_a_subclass_implements_is_not_a_stub() -> None:
    """The `CountVectorizer._weights` / `TfidfVectorizer._weights` shape."""
    found = _synthetic(
        '''
class Base:
    def _weights(self) -> Any:
        """The per-feature multiplier; none for plain counts."""
        return None


class Derived(Base):
    def _weights(self) -> Any:
        """The IDF vector."""
        import numpy as np

        return np.asarray(self.idf_)
'''
    )
    assert [f.category for f in silent.detect_stub(found)] == []


def test_a_documented_body_that_nobody_implements_is_still_a_stub() -> None:
    """The carve-out must not swallow the defect the rule exists for."""
    found = _synthetic(
        '''
class Base:
    def compute(self) -> int:
        """Return the computed total."""
        return None
'''
    )
    assert [f.category for f in silent.detect_stub(found)] == ["stub"]


def test_a_hook_whose_only_override_is_also_trivial_is_still_a_stub() -> None:
    """Two do-nothing bodies are not a template method; they are two unkept promises."""
    found = _synthetic(
        '''
class Base:
    def compute(self) -> int:
        """Return the computed total."""
        return None


class Derived(Base):
    def compute(self) -> int:
        """Return the computed total."""
        return None
'''
    )
    assert [f.category for f in silent.detect_stub(found)] == ["stub", "stub"]
