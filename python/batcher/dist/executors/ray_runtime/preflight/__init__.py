"""Worker binary-compatibility preflight: probe each node before it loads the engine.

`report` holds the facts and the comparison (pure, no Ray); `run` schedules the probe once
per node per Ray session and refuses a query whose shipped build a node cannot load.
"""

from __future__ import annotations

from .report import CompatibilityReport, Finding, PlatformFacts, compare
from .run import ensure_workers_compatible, last_compatibility_report, reset_preflight_cache

__all__ = [
    "CompatibilityReport",
    "Finding",
    "PlatformFacts",
    "compare",
    "ensure_workers_compatible",
    "last_compatibility_report",
    "reset_preflight_cache",
]
