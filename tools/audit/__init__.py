"""Codebase-health detectors — the mechanical half of the `audit-codebase-health` skill.

The registry below is the whole public surface: a name for each detector, mapping to a
function that takes the shared `Context` and yields `Finding`s. `tools/audit_health.py` is
the CLI over it. Adding a detector means adding a module beside these and one line here.
"""

from __future__ import annotations

from tools.audit.context import Context, Finding, build_context
from tools.audit.dead import detect_dead_python, detect_dead_rust
from tools.audit.production import detect_production
from tools.audit.silent import detect_near_duplicate, detect_stub, detect_swallowed
from tools.audit.testing import detect_test_quality

__all__ = ["DETECTORS", "SEVERITY_ORDER", "Context", "Finding", "build_context"]

#: Severity ordering for display. Findings are heuristics, so `high` means "almost certainly
#: worth acting on", not "proven".
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

DETECTORS = {
    "dead-python": detect_dead_python,
    "dead-rust": detect_dead_rust,
    "swallowed-error": detect_swallowed,
    "stub": detect_stub,
    "near-duplicate": detect_near_duplicate,
    "test-quality": detect_test_quality,
    "production": detect_production,
}
