"""Automatic findings for one run — what is wrong, the evidence, and what to do.

Rules are grouped by what they read — `resources` looks at the machine, `planning` at how work
was distributed, `dataflow` at row counts — because that grouping is also what decides whose
problem a finding is. `derive` holds the registry that runs them all and ranks the result.
"""

from __future__ import annotations

from batcher.observe.insights.derive import derive_insights

__all__ = ["derive_insights"]
